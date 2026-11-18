"""
Módulo de Backtesting com Horizonte Rolante Fora da Amostra (2024-2025 e Sintético) para o Artigo TEMPR.

Protocolo Metodológico (Horizonte Encolhendo + Curva a Termo Exógena Independente + Competência):
1. Curva a Termo de Mercado Independente (Major #1):
   - No backtest histórico (2024-2025): modelo econométrico AR(1) sazonal em log-PLD ajustado
     estritamente sem look-ahead até o mês 3q-1; todas as estratégias executam contra a mesma
     curva de mercado exógena (sem negociar contra a própria árvore).
   - No OOS sintético multi-trajetória (1.000 trajetórias): expectativa condicional analítica exata
     do DGP (bootstrap estacionário circular + saltos), independente da árvore Markoviana das estratégias.
2. Regras Regulatórias e Garantias (Major #3):
   - Audita insuficiência de lastro em janela móvel de 12 meses (Decreto 5.163/2004, VR CCEE 2025 = R$ 231,47/MWh,
     atualização pelo IPCA do VR base jul/2021 de R$ 189,90/MWh do Despacho ANEEL 3.176/2024, REN 1.090/2024)
     e pico de exposição spot curta em 2 meses consecutivos (proxy de chamadas de garantia CCEE),
     além do vale mensal de fluxo de caixa (Cash Min).
3. Calibração Econômica Realista (Major #4):
   - Tarifa média de venda ao cliente P_cli = R$ 160,00/MWh (margem realista no ACL), reportando
     resultados em Milhões de R$ e em R$/MWh.
4. Poder Estatístico e Separação Treino/Teste (Major #6 & #7):
   - Simulação paralela de 1.000 trajetórias sintéticas (ProcessPoolExecutor).
   - Suporte a DGP estritamente disjunto (pool 2021-2023) vs completo (2021-2025), bem como
     árvore diversificada 2001-2023 deflacionada pelo IPCA.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import math
import sys
import time
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
for sub in ("config", "optimization", "optimization/models", "data_processing", "utils"):
    sys.path.append(str(BASE / sub))

from arvore_ccee import carregar_base_mensal_expansiva, construir_arvore_ccee_treino
from arvore_cenarios import ArvoreCenarios, NoCenario
from modelo_multiestagio_tempr import ModeloMultiestagioTEMPR

OUTPUT_DIR = BASE / "resultados" / "backtesting_out_of_sample"
P_CLI_PADRAO = 160.0
VR_2025_PADRAO = 231.47


def carregar_dados_reais_teste():
    """Carrega dados mensais agregados do teste 2024-2025."""
    csv_teste = BASE / "data" / "processados_tempr" / "pld_horario_teste_2024_2025.csv"
    if not csv_teste.exists():
        csv_teste = BASE / "nyx" / "data" / "processados_tempr" / "pld_horario_teste_2024_2025.csv"
    df = pd.read_csv(csv_teste)

    grp = df.groupby(["ANO", "MES_REFERENCIA"])
    resumo = grp.agg({
        "PLD_SE": "mean",
        "PLD_NE": "mean",
        "HORA": "count",
    }).reset_index().rename(columns={"HORA": "horas"})
    resumo = resumo.sort_values("MES_REFERENCIA").reset_index(drop=True)
    return resumo


def _empacotar_curva_forward(
    pld_esp_se: np.ndarray,
    pld_esp_ne: np.ndarray,
    horas_rest: np.ndarray,
    spread: float,
) -> dict:
    """Converte vetores esperados de PLD spot (SE, NE) em cotações mensais, trimestrais e semestrais bid/ask."""
    n_rest = len(horas_rest)
    n_q = n_rest // 3
    n_s = n_rest // 6
    out = {}
    for s, p_esp in [("SE", pld_esp_se), ("NE", pld_esp_ne)]:
        pm_v = p_esp * (1.0 - spread)
        pm_c = p_esp * (1.0 + spread)
        pt_v = np.zeros(n_q, dtype=float)
        pt_c = np.zeros(n_q, dtype=float)
        for q_f in range(n_q):
            sl = slice(q_f * 3, q_f * 3 + 3)
            hq = horas_rest[sl]
            base_q = float(np.sum(p_esp[sl] * hq) / np.sum(hq))
            pt_v[q_f] = base_q * (1.0 - spread * 0.8)
            pt_c[q_f] = base_q * (1.0 + spread * 0.8)
        ps_v = np.zeros(n_s, dtype=float)
        ps_c = np.zeros(n_s, dtype=float)
        for h_f in range(n_s):
            sl = slice(h_f * 6, h_f * 6 + 6)
            hh = horas_rest[sl]
            base_h = float(np.sum(p_esp[sl] * hh) / np.sum(hh))
            ps_v[h_f] = base_h * (1.0 - spread * 0.6)
            ps_c[h_f] = base_h * (1.0 + spread * 0.6)
        out[s] = {
            "pld_esp": p_esp.copy(),
            "pm_v": pm_v, "pm_c": pm_c,
            "pt_v": pt_v, "pt_c": pt_c,
            "ps_v": ps_v, "ps_c": ps_c,
        }
    return out


def curva_termo_ar1_independente(
    mensal_hist_df: pd.DataFrame,
    m_start_abs: int,
    horas_rest: np.ndarray,
    spread: float = 0.01,
    ano_inicio: int = 2024,
) -> dict:
    """
    Major #1 (Histórico): Curva a termo econométrica AR(1) sazonal em log-PLD,
    estimada sem look-ahead sobre `mensal_hist_df` (até 3q-1) e independente da árvore Markoviana.
    """
    n_rest = len(horas_rest)
    df = mensal_hist_df.sort_index()
    meses_idx = df.index.to_numpy(dtype=int)
    cal_m = meses_idx % 100

    prev = {}
    for s, col in [("SE", "PLD_SE"), ("NE", "PLD_NE")]:
        y = np.log(np.maximum(df[col].to_numpy(dtype=float), 1.0))
        mu_global = float(np.mean(y))
        mu_cal = {}
        for mc in range(1, 13):
            vals_mc = y[cal_m == mc]
            # Contração Bayesiana leve caso haja poucas observações no mês calendário
            n_mc = len(vals_mc)
            mu_cal[mc] = float((np.sum(vals_mc) + 1.0 * mu_global) / (n_mc + 1.0))
        res = np.array([y[i] - mu_cal[int(cal_m[i])] for i in range(len(y))], dtype=float)
        if len(res) >= 4 and np.var(res[:-1]) > 1e-8:
            phi = float(np.clip(np.dot(res[1:], res[:-1]) / np.dot(res[:-1], res[:-1]), 0.0, 0.92))
        else:
            phi = 0.70
        innov = res[1:] - phi * res[:-1]
        sigma_e = float(np.std(innov, ddof=1)) if len(innov) > 2 else 0.25
        r_last = float(res[-1])

        p_esp = np.zeros(n_rest, dtype=float)
        for m_rel in range(n_rest):
            h = m_rel + 1
            m_abs = m_start_abs + m_rel
            mc_fut = (m_abs % 12) + 1
            ano_fut = ano_inicio + (m_abs // 12)
            lim = ModeloMultiestagioTEMPR.LIMITES_ANEEL.get(ano_fut, {"piso": 58.60, "teto_estrutural": 751.73})
            mean_log = mu_cal[mc_fut] + (phi ** h) * r_last
            var_h = (sigma_e ** 2) * (1.0 - phi ** (2 * h)) / max(1e-6, 1.0 - phi ** 2)
            p_raw = math.exp(mean_log + 0.5 * var_h)
            p_esp[m_rel] = float(np.clip(p_raw, lim["piso"], lim["teto_estrutural"]))
        prev[s] = p_esp

    return _empacotar_curva_forward(prev["SE"], prev["NE"], horas_rest, spread)


def _norm_ppf(p: float) -> float:
    """Aproximação de Beasley-Springer-Moro para quantil Normal padrão."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def _precomputar_expectativa_dgp(pld_pool_se: np.ndarray, pld_pool_ne: np.ndarray) -> dict:
    """
    Pré-computa a expectativa exata E[clip(pld_pool[i] * F_jump, piso_y, teto_y)]
    para o DGP (Politis-Romano + choques bivariados lognormais) em 2024 e 2025.
    """
    n_grid = 201
    zs = np.array([_norm_ppf((k + 0.5) / n_grid) for k in range(n_grid)], dtype=float)
    f_nojump = np.exp(0.06 * zs)
    f_jump_se = np.exp(0.25 + 0.25 * zs)
    f_jump_ne = np.exp(0.05 + 0.18 * zs)

    tabela = {"SE": {}, "NE": {}, "n_pool": len(pld_pool_se)}
    for ano in (2024, 2025):
        lim = ModeloMultiestagioTEMPR.LIMITES_ANEEL[ano]
        pmin, pmax = lim["piso"], lim["teto_estrutural"]
        # Matriz (n_pool, n_grid)
        nj_se = np.clip(pld_pool_se[:, None] * f_nojump[None, :], pmin, pmax).mean(axis=1)
        jp_se = np.clip(pld_pool_se[:, None] * f_jump_se[None, :], pmin, pmax).mean(axis=1)
        nj_ne = np.clip(pld_pool_ne[:, None] * f_nojump[None, :], pmin, pmax).mean(axis=1)
        jp_ne = np.clip(pld_pool_ne[:, None] * f_jump_ne[None, :], pmin, pmax).mean(axis=1)
        e_se = 0.88 * nj_se + 0.12 * jp_se
        e_ne = 0.88 * nj_ne + 0.12 * jp_ne
        tabela["SE"][ano] = {"por_idx": e_se, "media_estacionaria": float(np.mean(e_se))}
        tabela["NE"][ano] = {"por_idx": e_ne, "media_estacionaria": float(np.mean(e_ne))}
    return tabela


def curva_termo_dgp_analitica(
    tabela_dgp: dict,
    idx_last: int | None,
    m_start_abs: int,
    horas_rest: np.ndarray,
    spread: float = 0.01,
) -> dict:
    """
    Major #1 (Sintético): Expectativa condicional exata do DGP de bootstrap estacionário + saltos,
    dada apenas a posição realizada `idx_last` em 3q-1 (ou média estacionária em q=0).
    """
    n_pool = tabela_dgp["n_pool"]
    n_rest = len(horas_rest)
    p_se = np.zeros(n_rest, dtype=float)
    p_ne = np.zeros(n_rest, dtype=float)
    for m_rel in range(n_rest):
        m_abs = m_start_abs + m_rel
        ano = 2024 + (m_abs // 12)
        if idx_last is None:
            p_se[m_rel] = tabela_dgp["SE"][ano]["media_estacionaria"]
            p_ne[m_rel] = tabela_dgp["NE"][ano]["media_estacionaria"]
        else:
            h = m_rel + 1  # distância desde 3q-1
            w = (2.0 / 3.0) ** h
            idx_fut = (idx_last + h) % n_pool
            p_se[m_rel] = w * float(tabela_dgp["SE"][ano]["por_idx"][idx_fut]) + (1.0 - w) * tabela_dgp["SE"][ano]["media_estacionaria"]
            p_ne[m_rel] = w * float(tabela_dgp["NE"][ano]["por_idx"][idx_fut]) + (1.0 - w) * tabela_dgp["NE"][ano]["media_estacionaria"]
    return _empacotar_curva_forward(p_se, p_ne, horas_rest, spread)


def resolver_deterministico(
    arv,
    _lambda=0.5,
    spread=0.01,
    posicao_inicial=None,
    horas_mes=None,
    ano_inicio=2024,
    mes_inicio_calendario=0,
    preco_venda_cliente_mwh=P_CLI_PADRAO,
    penalidade_lastro_mwh=0.0,
    cobertura_minima_lastro=0.0,
):
    """Resolve o problema determinístico do cenário médio (Unhedged / DET)."""
    n_months = arv.n_months
    divisao_q = arv.divisao_estagios
    pld_medio_se = np.zeros(n_months)
    pld_medio_ne = np.zeros(n_months)
    dem_media = np.zeros(n_months)
    for leaf in arv.leaves:
        pld_medio_se += leaf.prob * leaf.pld_se
        pld_medio_ne += leaf.prob * leaf.pld_ne
        dem_media += leaf.prob * leaf.demanda_fator

    nodes_det = {}
    stages_det = [[i] for i in range(len(divisao_q))]
    for q_idx in range(len(divisao_q)):
        parent = (q_idx - 1) if q_idx > 0 else None
        nodes_det[q_idx] = NoCenario(
            id=q_idx, parent_id=parent, stage=q_idx, meses=divisao_q[q_idx], prob=1.0, cond_prob=1.0,
            cenarios_indices=[0], pld=pld_medio_se.copy(), pld_se=pld_medio_se.copy(),
            pld_ne=pld_medio_ne.copy(), demanda_fator=dem_media.copy(),
        )
        if parent is not None:
            nodes_det[parent].children_ids.append(q_idx)

    arvore_det = ArvoreCenarios(
        nodes=nodes_det, stages=stages_det, n_months=n_months, n_series_originais=1,
        divisao_estagios=divisao_q,
    )
    m_det = ModeloMultiestagioTEMPR(
        arvore_det,
        _lambda=_lambda,
        spread_termo=spread,
        permitir_revenda=False,
        posicao_inicial=posicao_inicial,
        kappa_demanda_max=1.0,
        horas_mes=horas_mes,
        ano_inicio=ano_inicio,
        mes_inicio_calendario=mes_inicio_calendario,
        preco_venda_cliente_mwh=preco_venda_cliente_mwh,
        penalidade_lastro_mwh=penalidade_lastro_mwh,
        cobertura_minima_lastro=cobertura_minima_lastro,
    )
    m_det.criar_variaveis()
    m_det.adicionar_restricoes_posicao_e_balanco()
    m_det.adicionar_restricoes_revenda()
    m_det.definir_funcao_objetivo()
    m_det.resolver(msg=False)
    return m_det


def _auditar_metricas_regulatorias_trajetoria(
    q_se: np.ndarray,
    q_ne: np.ndarray,
    d_se: np.ndarray,
    d_ne: np.ndarray,
    pld_se: np.ndarray,
    pld_ne: np.ndarray,
    horas: np.ndarray,
    vr_mwh: float = VR_2025_PADRAO,
) -> dict:
    """Calcula déficit de lastro (mensal e janela móvel 12m para m>=14), penalidade VR e exposição 2m CCEE."""
    bal_mw = (q_se + q_ne) - (d_se + d_ne)  # + sobra, - déficit
    # Déficit mensal em meses contratáveis m >= 3
    def_mensal_gwh = float(sum(max(0.0, -bal_mw[m]) * horas[m] for m in range(3, 24)) / 1e3)
    # Déficit em janela móvel de 12 meses nos meses totalmente contratáveis (m=14..23 => janela 3..14 .. 12..23)
    def_12m_mwh = 0.0
    meses_incump = 0
    for m in range(14, 24):
        avg_def = max(0.0, float(-np.mean(bal_mw[m - 11:m + 1])))
        if avg_def > 1e-3:
            meses_incump += 1
        def_12m_mwh += avg_def * float(horas[m])
    def_12m_gwh = float(def_12m_mwh / 1e3)
    penalidade_vr_M = float(def_12m_gwh * vr_mwh / 1e3)

    # Exposição curta mensal (spot short) e pico consecutivo de 2 meses (garantias CCEE)
    exp_m = np.maximum(0.0, d_se - q_se) * pld_se * horas + np.maximum(0.0, d_ne - q_ne) * pld_ne * horas
    pico_2m_M = float(max(exp_m[k] + exp_m[k + 1] for k in range(23)) / 1e6)
    return {
        "deficit_mensal_gwh": def_mensal_gwh,
        "deficit_12m_gwh": def_12m_gwh,
        "meses_incump_12m": meses_incump,
        "penalidade_vr_M": penalidade_vr_M,
        "garantia_pico_2m_M": pico_2m_M,
    }


def simular_horizonte_rolante(
    seed=2026,
    _lambda=0.5,
    spread=0.01,
    estrategias=None,
    gerador_diversificado=False,
    deflacionar_ipca=False,
    preco_venda_cliente_mwh=P_CLI_PADRAO,
    usar_curva_mercado_independente=True,
    penalidade_lastro_mwh=0.0,
    cobertura_minima_lastro=0.0,
):
    """
    Executa a simulação em horizonte rolante encolhendo até dez/2025 (24 meses).
    Quando `usar_curva_mercado_independente=True` (padrão), todas as transações são executadas
    contra uma curva a termo AR(1) sazonal independente estimada sem look-ahead.
    """
    dados_teste = carregar_dados_reais_teste()
    assert len(dados_teste) == 24, f"Esperado 24 meses de teste, obtido {len(dados_teste)}"

    if estrategias is None:
        estrategias = ["ME-R", "ME", "HEUR", "DET"]
    else:
        estrategias = list(estrategias)

    posicoes_herdadas = {
        est: {"SE": np.zeros(24, dtype=float), "NE": np.zeros(24, dtype=float)}
        for est in estrategias
    }
    custo_accrual = {est: np.zeros(24, dtype=float) for est in estrategias}
    mwh_contratados_accrual = {
        est: {"SE": np.zeros(24, dtype=float), "NE": np.zeros(24, dtype=float)}
        for est in estrategias
    }
    caixa_contratos_total = {est: 0.0 for est in estrategias}
    registros_mensais = []

    print(
        f"\n--- Backtest Rolante 2024-2025 (Seed={seed}, Lambda={_lambda}, P_cli=R${preco_venda_cliente_mwh:.0f}, "
        f"MktIndep={usar_curva_mercado_independente}) ---"
    )

    for q in range(8):
        m_start_abs = q * 3
        n_rest = 24 - m_start_abs
        n_stages_q = 8 - q
        divisao_q = [list(range(k * 3, k * 3 + 3)) for k in range(n_stages_q)]
        horas_rest = dados_teste["horas"].iloc[m_start_abs:24].to_numpy(dtype=float)

        meses_q = [m_start_abs, m_start_abs + 1, m_start_abs + 2]
        horas_q = [float(dados_teste.loc[m, "horas"]) for m in meses_q]
        pld_real_se_q = [float(dados_teste.loc[m, "PLD_SE"]) for m in meses_q]
        pld_real_ne_q = [float(dados_teste.loc[m, "PLD_NE"]) for m in meses_q]

        ultimo_mes_real = None if q == 0 else int(dados_teste.loc[m_start_abs - 1, "MES_REFERENCIA"])
        mensal_hist = carregar_base_mensal_expansiva(ultimo_mes_real)
        curva_mkt_q = (
            curva_termo_ar1_independente(mensal_hist, m_start_abs, horas_rest, spread=spread, ano_inicio=2024)
            if usar_curva_mercado_independente
            else None
        )

        arv = construir_arvore_ccee_treino(
            divisao_estagios=divisao_q,
            seed=seed + q * 17,
            volatilidade_demanda=0.08,
            correlacao_demanda_hidrologia=0.0,
            trimestre_inicial=q % 4,
            ultimo_mes_realizado=ultimo_mes_real,
            gerador_diversificado=gerador_diversificado,
            deflacionar_ipca=deflacionar_ipca,
        )

        for est in estrategias:
            pos_inicial_rel = {
                "SE": posicoes_herdadas[est]["SE"][m_start_abs:24].copy(),
                "NE": posicoes_herdadas[est]["NE"][m_start_abs:24].copy(),
            }

            if est in ("ME-R", "ME") or (q >= 1 and est in ("HEUR+R_resale", "HEUR+R_full")):
                m = ModeloMultiestagioTEMPR(
                    arvore=arv,
                    _lambda=_lambda,
                    spread_termo=spread,
                    permitir_revenda=(est != "ME"),
                    kappa_demanda_max=1.0,
                    posicao_inicial=pos_inicial_rel,
                    horas_mes=horas_rest,
                    ano_inicio=2024,
                    mes_inicio_calendario=m_start_abs,
                    preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                    penalidade_lastro_mwh=penalidade_lastro_mwh,
                    cobertura_minima_lastro=cobertura_minima_lastro,
                )
                m.criar_variaveis()
                m.adicionar_restricoes_posicao_e_balanco()
                m.adicionar_restricoes_revenda()
                if est == "HEUR+R_resale":
                    for k, v in m.variables.items():
                        if "_plus[" in k:
                            m.lp_model += (v == 0.0)
                m.definir_funcao_objetivo()
                m.resolver(msg=False)
            elif est == "DET":
                m = resolver_deterministico(
                    arv,
                    _lambda=_lambda,
                    spread=spread,
                    posicao_inicial=pos_inicial_rel,
                    horas_mes=horas_rest,
                    ano_inicio=2024,
                    mes_inicio_calendario=m_start_abs,
                    preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                    penalidade_lastro_mwh=penalidade_lastro_mwh,
                    cobertura_minima_lastro=cobertura_minima_lastro,
                )
            elif est == "HEUR" or (q == 0 and est in ("HEUR+R_resale", "HEUR+R_full")):
                m = None

            def _get_val(k: str) -> float:
                if m is None or k not in m.variables:
                    return 0.0
                v = m.variables[k].varValue
                return float(v) if v is not None else 0.0

            # Preços de execução de mercado no estágio q: curva exógena independente ou da própria árvore
            if usar_curva_mercado_independente:
                pfwd_c_m = {s: curva_mkt_q[s]["pm_c"] for s in ("SE", "NE")}
                pfwd_v_m = {s: curva_mkt_q[s]["pm_v"] for s in ("SE", "NE")}
                pfwd_c_t = {s: curva_mkt_q[s]["pt_c"] for s in ("SE", "NE")}
                pfwd_v_t = {s: curva_mkt_q[s]["pt_v"] for s in ("SE", "NE")}
                pfwd_c_s = {s: curva_mkt_q[s]["ps_c"] for s in ("SE", "NE")}
                pfwd_v_s = {s: curva_mkt_q[s]["ps_v"] for s in ("SE", "NE")}
            elif m is not None:
                pfwd_c_m = {s: m.precos_termo[0][s]["pm_c"] for s in ("SE", "NE")}
                pfwd_v_m = {s: m.precos_termo[0][s]["pm_v"] for s in ("SE", "NE")}
                pfwd_c_t = {s: m.precos_termo[0][s]["pt_c"] for s in ("SE", "NE")}
                pfwd_v_t = {s: m.precos_termo[0][s]["pt_v"] for s in ("SE", "NE")}
                pfwd_c_s = {s: m.precos_termo[0][s]["ps_c"] for s in ("SE", "NE")}
                pfwd_v_s = {s: m.precos_termo[0][s]["ps_v"] for s in ("SE", "NE")}
            else:
                m_tmp = ModeloMultiestagioTEMPR(
                    arvore=arv, _lambda=_lambda, spread_termo=spread, permitir_revenda=False,
                    horas_mes=horas_rest, ano_inicio=2024, mes_inicio_calendario=m_start_abs,
                    preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                )
                pfwd_c_m = {s: m_tmp.precos_termo[0][s]["pm_c"] for s in ("SE", "NE")}
                pfwd_v_m = {s: m_tmp.precos_termo[0][s]["pm_v"] for s in ("SE", "NE")}
                pfwd_c_t = pfwd_v_t = pfwd_c_s = pfwd_v_s = None

            fluxo_contratos_estagio_q = 0.0
            if est != "HEUR" and m is not None:
                for s in ("SE", "NE"):
                    for n_f in range(3, n_rest):
                        m_abs = m_start_abs + n_f
                        h_n = float(m.horas_mes[n_f])
                        xp = _get_val(f"xm_plus[0,{s},{n_f}]")
                        xm = _get_val(f"xm_minus[0,{s},{n_f}]")
                        custo_op = (pfwd_c_m[s][n_f] * xp - pfwd_v_m[s][n_f] * xm) * h_n
                        custo_accrual[est][m_abs] += custo_op
                        fluxo_contratos_estagio_q -= custo_op
                        mwh_contratados_accrual[est][s][m_abs] += (xp - xm) * h_n

                    for q_f in range(1, m.n_quarters):
                        st_m = q_f * 3
                        xp = _get_val(f"xt_plus[0,{s},{q_f}]")
                        xm = _get_val(f"xt_minus[0,{s},{q_f}]")
                        for offset in range(3):
                            n_rel = st_m + offset
                            m_abs = m_start_abs + n_rel
                            h_m_rel = float(m.horas_mes[n_rel])
                            custo_op = (pfwd_c_t[s][q_f] * xp - pfwd_v_t[s][q_f] * xm) * h_m_rel
                            custo_accrual[est][m_abs] += custo_op
                            fluxo_contratos_estagio_q -= custo_op
                            mwh_contratados_accrual[est][s][m_abs] += (xp - xm) * h_m_rel

                    for h_f in range(1, m.n_semesters):
                        st_m = h_f * 6
                        xp = _get_val(f"xs_plus[0,{s},{h_f}]")
                        xm = _get_val(f"xs_minus[0,{s},{h_f}]")
                        for offset in range(6):
                            n_rel = h_f * 6 + offset
                            m_abs = m_start_abs + n_rel
                            h_m_rel = float(m.horas_mes[n_rel])
                            custo_op = (pfwd_c_s[s][h_f] * xp - pfwd_v_s[s][h_f] * xm) * h_m_rel
                            custo_accrual[est][m_abs] += custo_op
                            fluxo_contratos_estagio_q -= custo_op
                            mwh_contratados_accrual[est][s][m_abs] += (xp - xm) * h_m_rel

            elif est.startswith("HEUR") and q == 0:
                for s in ("SE", "NE"):
                    d_b = 50.0 if s == "SE" else 20.0
                    for n_f in range(3, 24):
                        h_n = float(dados_teste.loc[n_f, "horas"])
                        vol_mw = 0.95 * d_b
                        custo_op = pfwd_c_m[s][n_f] * vol_mw * h_n
                        custo_accrual[est][n_f] += custo_op
                        fluxo_contratos_estagio_q -= custo_op
                        posicoes_herdadas[est][s][n_f] = vol_mw
                        mwh_contratados_accrual[est][s][n_f] += vol_mw * h_n

            caixa_contratos_total[est] += fluxo_contratos_estagio_q
            fluxo_contratos_por_mes = fluxo_contratos_estagio_q / 3.0

            for i_m, m_idx in enumerate(meses_q):
                h = horas_q[i_m]
                pld_se = pld_real_se_q[i_m]
                pld_ne = pld_real_ne_q[i_m]
                d_base_se = 50.0
                d_base_ne = 20.0

                q_final_se = float(posicoes_herdadas[est]["SE"][m_idx])
                q_final_ne = float(posicoes_herdadas[est]["NE"][m_idx])

                if m is not None:
                    q_raiz_se = _get_val(f"q[0,SE,{i_m}]")
                    q_raiz_ne = _get_val(f"q[0,NE,{i_m}]")
                    assert abs(q_final_se - q_raiz_se) < 1e-4, f"Mismatch q_final vs q_raiz SE: {q_final_se} vs {q_raiz_se}"
                    assert abs(q_final_ne - q_raiz_ne) < 1e-4, f"Mismatch q_final vs q_raiz NE: {q_final_ne} vs {q_raiz_ne}"

                assert abs(mwh_contratados_accrual[est]["SE"][m_idx] - q_final_se * h) < 1e-3
                assert abs(mwh_contratados_accrual[est]["NE"][m_idx] - q_final_ne * h) < 1e-3

                rec_clientes = (preco_venda_cliente_mwh * (d_base_se + d_base_ne)) * h
                encargos = (15.00 * (d_base_se + d_base_ne)) * h
                liq_spot = ((q_final_se - d_base_se) * pld_se + (q_final_ne - d_base_ne) * pld_ne) * h

                custo_contratos_acc_mes = float(custo_accrual[est][m_idx])
                lucro_accrual_mes = rec_clientes - encargos - custo_contratos_acc_mes + liq_spot
                fluxo_caixa_mes = rec_clientes - encargos + fluxo_contratos_por_mes + liq_spot

                registros_mensais.append({
                    "seed": seed,
                    "estrategia": est,
                    "trimestre": q,
                    "mes_absoluto": m_idx,
                    "mes_referencia": int(dados_teste.loc[m_idx, "MES_REFERENCIA"]),
                    "horas": h,
                    "PLD_SE": pld_se,
                    "PLD_NE": pld_ne,
                    "q_SE": q_final_se,
                    "q_NE": q_final_ne,
                    "mwh_entregue_SE": q_final_se * h,
                    "mwh_entregue_NE": q_final_ne * h,
                    "receita_clientes_M": rec_clientes / 1e6,
                    "encargos_M": encargos / 1e6,
                    "custo_contratos_M": custo_contratos_acc_mes / 1e6,
                    "custo_contratos_caixa_M": -fluxo_contratos_por_mes / 1e6,
                    "liquidacao_spot_M": liq_spot / 1e6,
                    "lucro_liquido_M": lucro_accrual_mes / 1e6,
                    "fluxo_caixa_M": fluxo_caixa_mes / 1e6,
                })

            if est != "HEUR" and m is not None:
                for n_fut in range(3, n_rest):
                    m_abs_fut = m_start_abs + n_fut
                    posicoes_herdadas[est]["SE"][m_abs_fut] = _get_val(f"q[0,SE,{n_fut}]")
                    posicoes_herdadas[est]["NE"][m_abs_fut] = _get_val(f"q[0,NE,{n_fut}]")

    df_mensal = pd.DataFrame(registros_mensais)
    for est in estrategias:
        sub = df_mensal[df_mensal["estrategia"] == est]
        tot_acc = float(sub["lucro_liquido_M"].sum())
        tot_cx = float(sub["fluxo_caixa_M"].sum())
        assert abs(tot_acc - tot_cx) < 1e-5, f"Divergência competência vs caixa em {est}: {tot_acc} vs {tot_cx}"

    return df_mensal


def bootstrap_diferenca_pareada(diffs: np.ndarray, n_boot=1000, bloco=3, seed=2026):
    """Calcula intervalo de confiança de 95% via bootstrap por blocos móveis sobre série de n meses."""
    rng = np.random.RandomState(seed)
    n = len(diffs)
    n_blocos = int(np.ceil(n / bloco))
    medias_boot = []
    for _ in range(n_boot):
        indices = []
        for _ in range(n_blocos):
            start = rng.randint(0, max(1, n - bloco + 1))
            indices.extend(range(start, min(n, start + bloco)))
        amostra = diffs[indices[:n]]
        medias_boot.append(float(np.mean(amostra)))
    return float(np.percentile(medias_boot, 2.5)), float(np.percentile(medias_boot, 97.5))


def executar_estudo_completo(
    n_seeds: int = 20,
    preco_venda_cliente_mwh: float = P_CLI_PADRAO,
    usar_curva_mercado_independente: bool = True,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [2024 + r for r in range(n_seeds)]
    estrategias = ["ME-R", "ME", "HEUR", "HEUR+R_resale", "HEUR+R_full", "DET"]
    todos_mensais = []
    resumos = []
    t0_global = time.time()

    for s in seeds:
        t0_s = time.time()
        df_m = simular_horizonte_rolante(
            seed=s,
            estrategias=estrategias,
            preco_venda_cliente_mwh=preco_venda_cliente_mwh,
            usar_curva_mercado_independente=usar_curva_mercado_independente,
        )
        t1_s = time.time()
        print(f"Seed {s} concluída em {t1_s - t0_s:.2f} s")
        todos_mensais.append(df_m)

        for est in estrategias:
            df_est = df_m[df_m["estrategia"] == est].sort_values("mes_absoluto")
            sub = df_est["lucro_liquido_M"].to_numpy()
            sub_cx = df_est["fluxo_caixa_M"].to_numpy()
            horas_a = df_est["horas"].to_numpy()
            mwh_tot_m = float(np.sum(70.0 * horas_a) / 1e6)
            reg_m = _auditar_metricas_regulatorias_trajetoria(
                q_se=df_est["q_SE"].to_numpy(),
                q_ne=df_est["q_NE"].to_numpy(),
                d_se=np.full(24, 50.0),
                d_ne=np.full(24, 20.0),
                pld_se=df_est["PLD_SE"].to_numpy(),
                pld_ne=df_est["PLD_NE"].to_numpy(),
                horas=horas_a,
            )
            lucro_tot = float(np.sum(sub))
            q05 = float(np.percentile(sub, 5))
            cvar05 = float(np.mean(sub[sub <= q05]))

            resumos.append({
                "seed": s,
                "estrategia": est,
                "lucro_total_M": lucro_tot,
                "lucro_r_mwh": lucro_tot / mwh_tot_m,
                "media_mensal_M": float(np.mean(sub)),
                "pior_mes_M": float(np.min(sub)),
                "pior_mes_caixa_M": float(np.min(sub_cx)),
                "CVaR_5pct_lucro_M": cvar05,
                **reg_m,
                "tempo_s": t1_s - t0_s,
            })

    tempo_total = time.time() - t0_global
    df_todos_mensais = pd.concat(todos_mensais, ignore_index=True)
    df_resumos = pd.DataFrame(resumos)

    df_todos_mensais.to_csv(OUTPUT_DIR / "backtesting_rolante_mensal_2024_2025.csv", index=False)
    df_resumos.to_csv(OUTPUT_DIR / "backtesting_rolante_resumo_2024_2025.csv", index=False)

    print("\n=======================================================")
    print(f"RESUMO DO BACKTESTING (2024-2025, {n_seeds} SEEDS, P_cli=R${preco_venda_cliente_mwh:.0f}, MktIndep={usar_curva_mercado_independente}):")
    cols_grp = [
        "lucro_total_M", "lucro_r_mwh", "media_mensal_M", "pior_mes_M",
        "pior_mes_caixa_M", "CVaR_5pct_lucro_M", "deficit_12m_gwh", "penalidade_vr_M", "garantia_pico_2m_M",
    ]
    resumo_medio = df_resumos.groupby("estrategia")[cols_grp].mean()
    resumo_std = df_resumos.groupby("estrategia")[["lucro_total_M", "CVaR_5pct_lucro_M"]].std(ddof=1)
    print(resumo_medio.round(3))

    ics_info = {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "preco_venda_cliente_mwh": preco_venda_cliente_mwh,
        "usar_curva_mercado_independente": usar_curva_mercado_independente,
        "resumo_medio_seeds": resumo_medio.round(6).to_dict(orient="index"),
        "resumo_std_seeds": resumo_std.round(6).to_dict(orient="index"),
        "por_seed": {},
        "media_seeds_por_mes": {},
        "estatisticas_entre_seeds": {},
    }

    pares = [
        ("ME-R_vs_ME", "ME-R", "ME"),
        ("ME-R_vs_HEUR", "ME-R", "HEUR"),
        ("ME_vs_DET", "ME", "DET"),
        ("HEUR_vs_ME", "HEUR", "ME"),
        ("HEUR_vs_ME-R", "HEUR", "ME-R"),
        ("HEUR+R_resale_vs_HEUR", "HEUR+R_resale", "HEUR"),
        ("HEUR+R_full_vs_HEUR", "HEUR+R_full", "HEUR"),
        ("HEUR+R_full_vs_ME-R", "HEUR+R_full", "ME-R"),
    ]

    for s in seeds:
        df_s = df_todos_mensais[df_todos_mensais["seed"] == s].sort_values("mes_absoluto")
        ics_info["por_seed"][str(s)] = {}
        for nome_par, e1, e2 in pares:
            v1 = df_s[df_s["estrategia"] == e1]["lucro_liquido_M"].to_numpy()
            v2 = df_s[df_s["estrategia"] == e2]["lucro_liquido_M"].to_numpy()
            diff = v1 - v2
            ic = bootstrap_diferenca_pareada(diff, seed=2026)
            ics_info["por_seed"][str(s)][nome_par] = {
                "delta_total_M": float(np.sum(diff)),
                "media_mensal_M": float(np.mean(diff)),
                "ic95_inf_M": ic[0],
                "ic95_sup_M": ic[1],
            }

    piv_tot = df_resumos.pivot(index="seed", columns="estrategia", values="lucro_total_M")
    for nome_par, e1, e2 in pares:
        d_tot = (piv_tot[e1] - piv_tot[e2]).to_numpy(dtype=float)
        ics_info["estatisticas_entre_seeds"][nome_par] = {
            "media_delta_total_M": float(np.mean(d_tot)),
            "std_delta_total_M": float(np.std(d_tot, ddof=1)),
            "min_delta_total_M": float(np.min(d_tot)),
            "max_delta_total_M": float(np.max(d_tot)),
            "n_seeds_positivo": int(np.sum(d_tot > 0.0)),
            "n_seeds_ic_exclui_zero": int(sum(1 for s in seeds if ics_info["por_seed"][str(s)][nome_par]["ic95_inf_M"] > 0.0)),
        }

    df_media_mes = (
        df_todos_mensais.groupby(["estrategia", "mes_absoluto"], as_index=False)["lucro_liquido_M"]
        .mean()
        .sort_values("mes_absoluto")
    )

    print(f"\n--- Comparações Pareadas na Média das {n_seeds} Seeds (n=24 meses, Bloco=3, B=1000) ---")
    for nome_par, e1, e2 in pares:
        v1 = df_media_mes[df_media_mes["estrategia"] == e1]["lucro_liquido_M"].to_numpy()
        v2 = df_media_mes[df_media_mes["estrategia"] == e2]["lucro_liquido_M"].to_numpy()
        diff = v1 - v2
        ic = bootstrap_diferenca_pareada(diff, seed=2026)
        ics_info["media_seeds_por_mes"][nome_par] = {
            "delta_total_M": float(np.sum(diff)),
            "media_mensal_M": float(np.mean(diff)),
            "ic95_inf_M": ic[0],
            "ic95_sup_M": ic[1],
        }
        st_s = ics_info["estatisticas_entre_seeds"][nome_par]
        print(
            f"{nome_par:22s}: Delta Total = {np.sum(diff):+.2f}M (SD_seeds={st_s['std_delta_total_M']:.2f}M, "
            f"pos={st_s['n_seeds_positivo']}/{n_seeds}) | Media = {np.mean(diff):+.3f}M [IC 95%: {ic[0]:+.3f}, {ic[1]:+.3f}]"
        )

    with open(OUTPUT_DIR / "backtesting_rolante_ics_2024_2025.json", "w", encoding="utf-8") as f:
        json.dump(ics_info, f, indent=4, ensure_ascii=False)
    print(f"Tempo Total (Histórico {n_seeds} seeds): {tempo_total:.2f} s")
    return df_resumos


def _gerar_trajetorias_sinteticas_independentes(
    n_trajetorias: int = 1000,
    seed_dgp: int = 7777,
    pool_dgp: str = "2021_2023",
):
    """
    Gera trajetórias sintéticas de 24 meses por um DGP independente da árvore Markoviana:
    - pool_dgp="2021_2023" (Major #6): reamostra estritamente os 36 meses pré-teste (sem vazamento de 2024-2025).
    - pool_dgp="2021_2025": inclui todos os 60 meses empíricos (análise de sensibilidade).
    Retorna `(trajetorias, tabela_dgp)` com índices do pool para a curva condicional exata do DGP.
    """
    csv_tr = BASE / "data" / "processados_tempr" / "pld_horario_treino_2021_2023.csv"
    csv_te = BASE / "data" / "processados_tempr" / "pld_horario_teste_2024_2025.csv"
    df_tr = pd.read_csv(csv_tr).groupby("MES_REFERENCIA")[["PLD_SE", "PLD_NE"]].mean()
    df_te_raw = pd.read_csv(csv_te)
    df_te = df_te_raw.groupby("MES_REFERENCIA").agg(
        PLD_SE=("PLD_SE", "mean"), PLD_NE=("PLD_NE", "mean"), horas=("HORA", "count"),
    )

    if pool_dgp == "2021_2023":
        pool = df_tr.sort_index()
    elif pool_dgp == "2021_2025":
        pool = pd.concat([df_tr, df_te[["PLD_SE", "PLD_NE"]]]).sort_index()
    else:
        raise ValueError(f"pool_dgp inválido: {pool_dgp}")

    pld_pool_se = pool["PLD_SE"].to_numpy(dtype=float)
    pld_pool_ne = pool["PLD_NE"].to_numpy(dtype=float)
    n_pool = len(pld_pool_se)
    horas_24 = df_te.sort_index()["horas"].to_numpy(dtype=float)
    meses_ref_24 = df_te.sort_index().index.to_numpy(dtype=int)
    tabela_dgp = _precomputar_expectativa_dgp(pld_pool_se, pld_pool_ne)

    rng = np.random.RandomState(seed_dgp)
    trajetorias = []
    for k in range(n_trajetorias):
        se_out = np.zeros(24, dtype=float)
        ne_out = np.zeros(24, dtype=float)
        dem_fator = np.zeros(24, dtype=float)
        pool_idxs = np.zeros(24, dtype=int)

        idx = rng.randint(0, n_pool)
        theta_prev = 1.0
        for m in range(24):
            if m > 0:
                if rng.rand() < (1.0 / 3.0):
                    idx = rng.randint(0, n_pool)
                else:
                    idx = (idx + 1) % n_pool
            pool_idxs[m] = idx
            base_se = pld_pool_se[idx]
            base_ne = pld_pool_ne[idx]
            if m % 3 == 0:
                if rng.rand() < 0.12:
                    fator_se_q = float(np.exp(rng.normal(0.25, 0.25)))
                    fator_ne_q = float(np.exp(rng.normal(0.05, 0.18)))
                else:
                    fator_se_q = float(np.exp(rng.normal(0.0, 0.06)))
                    fator_ne_q = float(np.exp(rng.normal(0.0, 0.06)))
            ano = 2024 + (m // 12)
            lim = ModeloMultiestagioTEMPR.LIMITES_ANEEL[ano]
            se_out[m] = np.clip(base_se * fator_se_q, lim["piso"], lim["teto_estrutural"])
            ne_out[m] = np.clip(base_ne * fator_ne_q, lim["piso"], lim["teto_estrutural"])
            theta_prev = float(np.clip(1.0 + 0.75 * (theta_prev - 1.0) + rng.normal(0.0, 0.055), 0.75, 1.25))
            dem_fator[m] = theta_prev

        trajetorias.append({
            "traj_id": k,
            "meses_ref": meses_ref_24,
            "horas": horas_24,
            "PLD_SE": se_out,
            "PLD_NE": ne_out,
            "dem_fator": dem_fator,
            "pool_idxs": pool_idxs,
        })
    return trajetorias, tabela_dgp


def _resolver_q0_payload(
    seed_arvore: int,
    _lambda: float,
    spread: float,
    gerador_diversificado: bool,
    deflacionar_ipca: bool,
    preco_venda_cliente_mwh: float,
    horas_24: np.ndarray,
    tabela_dgp: dict,
    usar_curva_mercado_independente: bool,
) -> dict:
    """Resolve q=0 uma única vez e serializa decisões + preços a termo para os workers paralelos."""
    divisao_q0 = [list(range(k * 3, k * 3 + 3)) for k in range(8)]
    arv_q0 = construir_arvore_ccee_treino(
        divisao_estagios=divisao_q0, seed=seed_arvore, volatilidade_demanda=0.08,
        correlacao_demanda_hidrologia=0.0, trimestre_inicial=0, ultimo_mes_realizado=None,
        gerador_diversificado=gerador_diversificado, deflacionar_ipca=deflacionar_ipca,
    )
    mkt_q0 = curva_termo_dgp_analitica(tabela_dgp, idx_last=None, m_start_abs=0, horas_rest=horas_24, spread=spread)
    payload = {"mkt_q0": mkt_q0, "estrategias": {}}

    for est in ("ME-R", "ME", "DET"):
        if est == "DET":
            m0 = resolver_deterministico(
                arv_q0, _lambda=_lambda, spread=spread, horas_mes=horas_24, ano_inicio=2024,
                mes_inicio_calendario=0, preco_venda_cliente_mwh=preco_venda_cliente_mwh,
            )
        else:
            m0 = ModeloMultiestagioTEMPR(
                arvore=arv_q0, _lambda=_lambda, spread_termo=spread, permitir_revenda=(est == "ME-R"),
                kappa_demanda_max=1.0, horas_mes=horas_24, ano_inicio=2024, mes_inicio_calendario=0,
                preco_venda_cliente_mwh=preco_venda_cliente_mwh,
            )
            m0.criar_variaveis()
            m0.adicionar_restricoes_posicao_e_balanco()
            m0.adicionar_restricoes_revenda()
            m0.definir_funcao_objetivo()
            m0.resolver(msg=False)

        def _gv0(name: str) -> float:
            v = m0.variables.get(name)
            return float(v.varValue) if (v is not None and v.varValue is not None) else 0.0

        trades = {}
        for s in ("SE", "NE"):
            trades[s] = {
                "xm_p": {n: _gv0(f"xm_plus[0,{s},{n}]") for n in range(3, 24)},
                "xm_m": {n: _gv0(f"xm_minus[0,{s},{n}]") for n in range(3, 24)},
                "xt_p": {qf: _gv0(f"xt_plus[0,{s},{qf}]") for qf in range(1, m0.n_quarters)},
                "xt_m": {qf: _gv0(f"xt_minus[0,{s},{qf}]") for qf in range(1, m0.n_quarters)},
                "xs_p": {hf: _gv0(f"xs_plus[0,{s},{hf}]") for hf in range(1, m0.n_semesters)},
                "xs_m": {hf: _gv0(f"xs_minus[0,{s},{hf}]") for hf in range(1, m0.n_semesters)},
                "q_fut": {nf: _gv0(f"q[0,{s},{nf}]") for nf in range(3, 24)},
            }
        tree_p = {s: {k: v.copy() for k, v in m0.precos_termo[0][s].items()} for s in ("SE", "NE")}
        payload["estrategias"][est] = {
            "trades": trades,
            "precos": mkt_q0 if usar_curva_mercado_independente else tree_p,
        }
    return payload


def _worker_oos_trajetoria(task_args: tuple) -> list[dict]:
    """Worker paralelo para simular uma trajetória sintética nos 8 trimestres rolantes."""
    (
        tr, seed_arvore, _lambda, spread, gerador_diversificado, deflacionar_ipca,
        preco_venda_cliente_mwh, usar_curva_mercado_independente, q0_payload, tabela_dgp, estrategias,
    ) = task_args

    k = tr["traj_id"]
    pld_se_path = tr["PLD_SE"]
    pld_ne_path = tr["PLD_NE"]
    dem_path = tr["dem_fator"]
    meses_ref = tr["meses_ref"]
    horas_24 = tr["horas"]
    pool_idxs = tr["pool_idxs"]
    df_extra = pd.DataFrame({"PLD_SE": pld_se_path, "PLD_NE": pld_ne_path}, index=meses_ref)

    posicoes_herdadas = {est: {"SE": np.zeros(24), "NE": np.zeros(24)} for est in estrategias}
    custo_accrual = {est: np.zeros(24) for est in estrategias}

    # q = 0 (usa decisões cacheadas)
    for est in estrategias:
        if est.startswith("HEUR"):
            pm_c = q0_payload["mkt_q0"]
            for s in ("SE", "NE"):
                d_b = 50.0 if s == "SE" else 20.0
                for n_f in range(3, 24):
                    vol_mw = 0.95 * d_b
                    custo_accrual[est][n_f] += pm_c[s]["pm_c"][n_f] * vol_mw * float(horas_24[n_f])
                    posicoes_herdadas[est][s][n_f] = vol_mw
        else:
            info = q0_payload["estrategias"][est]
            trd = info["trades"]
            prc = info["precos"]
            for s in ("SE", "NE"):
                for n_f, xp in trd[s]["xm_p"].items():
                    xm = trd[s]["xm_m"][n_f]
                    custo_accrual[est][n_f] += (prc[s]["pm_c"][n_f] * xp - prc[s]["pm_v"][n_f] * xm) * float(horas_24[n_f])
                for q_f, xp in trd[s]["xt_p"].items():
                    xm = trd[s]["xt_m"][q_f]
                    for off in range(3):
                        nr = q_f * 3 + off
                        custo_accrual[est][nr] += (prc[s]["pt_c"][q_f] * xp - prc[s]["pt_v"][q_f] * xm) * float(horas_24[nr])
                for h_f, xp in trd[s]["xs_p"].items():
                    xm = trd[s]["xs_m"][h_f]
                    for off in range(6):
                        nr = h_f * 6 + off
                        custo_accrual[est][nr] += (prc[s]["ps_c"][h_f] * xp - prc[s]["ps_v"][h_f] * xm) * float(horas_24[nr])
                for n_f, qv in trd[s]["q_fut"].items():
                    posicoes_herdadas[est][s][n_f] = qv

    # q = 1..7
    for q in range(1, 8):
        m_start_abs = q * 3
        n_rest = 24 - m_start_abs
        divisao_q = [list(range(idx * 3, idx * 3 + 3)) for idx in range(8 - q)]
        horas_rest = horas_24[m_start_abs:24]
        ultimo_mes_real = int(meses_ref[m_start_abs - 1])

        arv_q = construir_arvore_ccee_treino(
            divisao_estagios=divisao_q, seed=seed_arvore + q * 17,
            volatilidade_demanda=0.08, correlacao_demanda_hidrologia=0.0,
            trimestre_inicial=q % 4, ultimo_mes_realizado=ultimo_mes_real,
            mensal_extra_df=df_extra, gerador_diversificado=gerador_diversificado,
            deflacionar_ipca=deflacionar_ipca,
        )
        curva_mkt_q = (
            curva_termo_dgp_analitica(
                tabela_dgp, idx_last=int(pool_idxs[m_start_abs - 1]),
                m_start_abs=m_start_abs, horas_rest=horas_rest, spread=spread,
            )
            if usar_curva_mercado_independente
            else None
        )

        for est in estrategias:
            if est == "HEUR":
                continue
            pos_inicial_rel = {
                "SE": posicoes_herdadas[est]["SE"][m_start_abs:24].copy(),
                "NE": posicoes_herdadas[est]["NE"][m_start_abs:24].copy(),
            }
            if est == "DET":
                m = resolver_deterministico(
                    arv_q, _lambda=_lambda, spread=spread, posicao_inicial=pos_inicial_rel,
                    horas_mes=horas_rest, ano_inicio=2024, mes_inicio_calendario=m_start_abs,
                    preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                )
            else:
                m = ModeloMultiestagioTEMPR(
                    arvore=arv_q, _lambda=_lambda, spread_termo=spread, permitir_revenda=(est != "ME"),
                    kappa_demanda_max=1.0, posicao_inicial=pos_inicial_rel,
                    horas_mes=horas_rest, ano_inicio=2024, mes_inicio_calendario=m_start_abs,
                    preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                )
                m.criar_variaveis()
                m.adicionar_restricoes_posicao_e_balanco()
                m.adicionar_restricoes_revenda()
                if est == "HEUR+R_resale":
                    for vname, var in m.variables.items():
                        if "_plus[" in vname:
                            m.lp_model += (var == 0.0)
                m.definir_funcao_objetivo()
                m.resolver(msg=False)

            def _gv(name: str) -> float:
                v = m.variables.get(name)
                return float(v.varValue) if (v is not None and v.varValue is not None) else 0.0

            prc_q = curva_mkt_q if usar_curva_mercado_independente else m.precos_termo[0]
            for s in ("SE", "NE"):
                pt_s = prc_q[s]
                for n_f in range(3, n_rest):
                    m_abs = m_start_abs + n_f
                    h_n = float(m.horas_mes[n_f])
                    xp = _gv(f"xm_plus[0,{s},{n_f}]")
                    xm = _gv(f"xm_minus[0,{s},{n_f}]")
                    custo_accrual[est][m_abs] += (pt_s["pm_c"][n_f] * xp - pt_s["pm_v"][n_f] * xm) * h_n
                for q_f in range(1, m.n_quarters):
                    xp = _gv(f"xt_plus[0,{s},{q_f}]")
                    xm = _gv(f"xt_minus[0,{s},{q_f}]")
                    for offset in range(3):
                        n_rel = q_f * 3 + offset
                        h_m_rel = float(m.horas_mes[n_rel])
                        custo_accrual[est][m_start_abs + n_rel] += (pt_s["pt_c"][q_f] * xp - pt_s["pt_v"][q_f] * xm) * h_m_rel
                for h_f in range(1, m.n_semesters):
                    xp = _gv(f"xs_plus[0,{s},{h_f}]")
                    xm = _gv(f"xs_minus[0,{s},{h_f}]")
                    for offset in range(6):
                        n_rel = h_f * 6 + offset
                        h_m_rel = float(m.horas_mes[n_rel])
                        custo_accrual[est][m_start_abs + n_rel] += (pt_s["ps_c"][h_f] * xp - pt_s["ps_v"][h_f] * xm) * h_m_rel
            for n_fut in range(3, n_rest):
                posicoes_herdadas[est]["SE"][m_start_abs + n_fut] = _gv(f"q[0,SE,{n_fut}]")
                posicoes_herdadas[est]["NE"][m_start_abs + n_fut] = _gv(f"q[0,NE,{n_fut}]")

    out_rows = []
    d_se_vec = 50.0 * dem_path
    d_ne_vec = 20.0 * dem_path
    mwh_tot_m = float(np.sum((d_se_vec + d_ne_vec) * horas_24) / 1e6)

    for est in estrategias:
        lucro_meses = np.zeros(24, dtype=float)
        for m_idx in range(24):
            h = float(horas_24[m_idx])
            d_se = float(d_se_vec[m_idx])
            d_ne = float(d_ne_vec[m_idx])
            rec = preco_venda_cliente_mwh * (d_se + d_ne) * h
            enc = 15.0 * (d_se + d_ne) * h
            spot = ((posicoes_herdadas[est]["SE"][m_idx] - d_se) * pld_se_path[m_idx] +
                    (posicoes_herdadas[est]["NE"][m_idx] - d_ne) * pld_ne_path[m_idx]) * h
            lucro_meses[m_idx] = (rec - enc - custo_accrual[est][m_idx] + spot) / 1e6

        reg_m = _auditar_metricas_regulatorias_trajetoria(
            q_se=posicoes_herdadas[est]["SE"],
            q_ne=posicoes_herdadas[est]["NE"],
            d_se=d_se_vec,
            d_ne=d_ne_vec,
            pld_se=pld_se_path,
            pld_ne=pld_ne_path,
            horas=horas_24,
        )
        lt = float(np.sum(lucro_meses))
        out_rows.append({
            "traj_id": k,
            "estrategia": est,
            "lucro_total_M": lt,
            "lucro_r_mwh": lt / mwh_tot_m,
            "pior_mes_M": float(np.min(lucro_meses)),
            "media_mensal_M": float(np.mean(lucro_meses)),
            **reg_m,
        })
    return out_rows


def executar_oos_sintetico_multipath(
    n_trajetorias: int = 1000,
    seed_arvore: int = 2026,
    _lambda: float = 0.5,
    spread: float = 0.01,
    gerador_diversificado: bool = False,
    deflacionar_ipca: bool = False,
    pool_dgp: str = "2021_2023",
    preco_venda_cliente_mwh: float = P_CLI_PADRAO,
    usar_curva_mercado_independente: bool = True,
    n_workers: int = 16,
    tag_arquivo: str | None = None,
):
    """
    Avalia ME-R, ME, HEUR, HEUR+R_resale, HEUR+R_full e DET fora da amostra em `n_trajetorias`
    trajetórias sintéticas geradas por um DGP independente, com execução paralela.
    """
    from estabilidade_cenarios import _t_crit_95

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajs, tabela_dgp = _gerar_trajetorias_sinteticas_independentes(
        n_trajetorias=n_trajetorias, pool_dgp=pool_dgp,
    )
    estrategias = ["ME-R", "ME", "HEUR", "HEUR+R_resale", "HEUR+R_full", "DET"]
    horas_24 = trajs[0]["horas"]

    q0_payload = _resolver_q0_payload(
        seed_arvore=seed_arvore,
        _lambda=_lambda,
        spread=spread,
        gerador_diversificado=gerador_diversificado,
        deflacionar_ipca=deflacionar_ipca,
        preco_venda_cliente_mwh=preco_venda_cliente_mwh,
        horas_24=horas_24,
        tabela_dgp=tabela_dgp,
        usar_curva_mercado_independente=usar_curva_mercado_independente,
    )

    t0_sim = time.time()
    tasks = [
        (
            tr, seed_arvore, _lambda, spread, gerador_diversificado, deflacionar_ipca,
            preco_venda_cliente_mwh, usar_curva_mercado_independente, q0_payload, tabela_dgp, estrategias,
        )
        for tr in trajs
    ]

    registros_traj = []
    if n_workers > 1 and n_trajetorias >= 20:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for chunk in ex.map(_worker_oos_trajetoria, tasks, chunksize=8):
                registros_traj.extend(chunk)
    else:
        for tsk in tasks:
            registros_traj.extend(_worker_oos_trajetoria(tsk))

    df_sin = pd.DataFrame(registros_traj)
    if tag_arquivo is None:
        suf_gen = "div_ipca" if (gerador_diversificado and deflacionar_ipca) else ("diversificado" if gerador_diversificado else "base")
        tag_arquivo = f"{suf_gen}_pool{pool_dgp}_{n_trajetorias}"
    csv_name = f"oos_sintetico_{tag_arquivo}.csv"
    json_name = f"oos_sintetico_{tag_arquivo}_resumo.json"
    df_sin.to_csv(OUTPUT_DIR / csv_name, index=False)

    piv = df_sin.pivot(index="traj_id", columns="estrategia", values="lucro_total_M")
    piv_mwh = df_sin.pivot(index="traj_id", columns="estrategia", values="lucro_r_mwh")
    t_crit = _t_crit_95(n_trajetorias - 1)
    resumo_sin = {
        "n_trajetorias": n_trajetorias,
        "pool_dgp": pool_dgp,
        "gerador_diversificado": gerador_diversificado,
        "deflacionar_ipca": deflacionar_ipca,
        "preco_venda_cliente_mwh": preco_venda_cliente_mwh,
        "usar_curva_mercado_independente": usar_curva_mercado_independente,
        "estrategias": {},
        "comparacoes_pareadas": {},
    }

    print(
        f"\n=== OOS SINTÉTICO ({tag_arquivo}, N={n_trajetorias}, P_cli=R${preco_venda_cliente_mwh:.0f}, "
        f"MktIndep={usar_curva_mercado_independente}) ==="
    )
    for est in estrategias:
        vals = piv[est].to_numpy(dtype=float)
        sub_df = df_sin[df_sin["estrategia"] == est]
        q05 = float(np.percentile(vals, 5))
        cvar05 = float(np.mean(vals[vals <= q05]))
        pior_m_est = float(sub_df["pior_mes_M"].mean())
        def_12m_m = float(sub_df["deficit_12m_gwh"].mean())
        pen_vr_m = float(sub_df["penalidade_vr_M"].mean())
        gar_2m_p95 = float(np.percentile(sub_df["garantia_pico_2m_M"], 95))
        resumo_sin["estrategias"][est] = {
            "media_lucro_total_M": float(np.mean(vals)),
            "std_lucro_total_M": float(np.std(vals, ddof=1)),
            "media_lucro_r_mwh": float(piv_mwh[est].mean()),
            "cvar5_lucro_total_M": cvar05,
            "pior_mes_medio_M": pior_m_est,
            "deficit_12m_medio_gwh": def_12m_m,
            "penalidade_vr_media_M": pen_vr_m,
            "garantia_pico_2m_p95_M": gar_2m_p95,
        }
        print(
            f"  {est:13s}: E[R]={np.mean(vals):6.2f}M +- {np.std(vals, ddof=1):5.2f}M ({piv_mwh[est].mean():5.2f} R$/MWh)"
            f" | CVaR5%={cvar05:7.2f}M | Def12M={def_12m_m:5.1f}GWh (VR={pen_vr_m:5.2f}M) | Gar95={gar_2m_p95:5.2f}M"
        )

    pares_sin = [
        ("ME-R_vs_ME", "ME-R", "ME"),
        ("ME-R_vs_HEUR", "ME-R", "HEUR"),
        ("HEUR_vs_ME-R", "HEUR", "ME-R"),
        ("ME_vs_DET", "ME", "DET"),
        ("HEUR_vs_ME", "HEUR", "ME"),
        ("HEUR+R_resale_vs_HEUR", "HEUR+R_resale", "HEUR"),
        ("HEUR+R_full_vs_HEUR", "HEUR+R_full", "HEUR"),
        ("HEUR+R_full_vs_ME-R", "HEUR+R_full", "ME-R"),
    ]
    for nome_par, e1, e2 in pares_sin:
        diff = (piv[e1] - piv[e2]).to_numpy(dtype=float)
        diff_mwh = (piv_mwh[e1] - piv_mwh[e2]).to_numpy(dtype=float)
        mean_d = float(np.mean(diff))
        sd_d = float(np.std(diff, ddof=1))
        se_d = sd_d / np.sqrt(n_trajetorias)
        ic_l, ic_u = mean_d - t_crit * se_d, mean_d + t_crit * se_d
        n_pos = int(np.sum(diff > 0.0))
        resumo_sin["comparacoes_pareadas"][nome_par] = {
            "media_delta_M": mean_d,
            "std_delta_M": sd_d,
            "media_delta_r_mwh": float(np.mean(diff_mwh)),
            "ic95_inf_M": ic_l,
            "ic95_sup_M": ic_u,
            "n_traj_positivo": n_pos,
        }
        print(
            f"  {nome_par:21s}: Delta={mean_d:+.2f}M +- {sd_d:.2f}M ({np.mean(diff_mwh):+.2f} R$/MWh)"
            f" | IC95%=[{ic_l:+.2f}, {ic_u:+.2f}]M | win={n_pos}/{n_trajetorias}"
        )

    with open(OUTPUT_DIR / json_name, "w", encoding="utf-8") as f:
        json.dump(resumo_sin, f, indent=4, ensure_ascii=False)
    print(f"Simulação OOS sintética ({tag_arquivo}) concluída em {time.time() - t0_sim:.2f} s.")
    return df_sin, resumo_sin


if __name__ == "__main__":
    executar_estudo_completo(n_seeds=20)
    executar_oos_sintetico_multipath(
        n_trajetorias=1000, pool_dgp="2021_2023", gerador_diversificado=False, deflacionar_ipca=False,
    )
    executar_oos_sintetico_multipath(
        n_trajetorias=1000, pool_dgp="2021_2023", gerador_diversificado=True, deflacionar_ipca=True,
    )
    executar_oos_sintetico_multipath(
        n_trajetorias=1000, pool_dgp="2021_2025", gerador_diversificado=False, deflacionar_ipca=False,
    )
