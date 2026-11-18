"""
Construtor de árvore de cenários estritamente calibrada com os dados reais da CCEE.
- Carrega pld_horario_treino_2021_2023.csv e, em janela expansiva fora da amostra,
  incorpora os meses realizados de pld_horario_teste_2024_2025.csv estritamente até ultimo_mes_realizado.
- Agrega em blocos trimestrais reais com alinhamento sazonal (trimestre_inicial) e
  atualização da matriz de transição Markoviana e do estado inicial condicionado.
- Inclui séries reais de PLD_SE e PLD_NE por patamar/mês.
- Gera incerteza de demanda calibrada por nó (demanda_fator) representando migração/perda de clientes.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

from arvore_cenarios import ArvoreCenarios, NoCenario

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "nyx" / "data" / "processados_tempr"

# Contagens brutas de transição de regimes trimestrais na série histórica CCEE 2001-2023 (91 transições)
_T_BASE_2001_2023 = np.array([
    [15.0,  1.0,  9.0],  # de 0 (seco > 210 nominal)
    [ 0.0, 28.0,  7.0],  # de 1 (úmido < 68 nominal)
    [10.0,  6.0, 15.0],  # de 2 (normal 68-210 nominal)
], dtype=float)

# Taxas anuais oficiais do IPCA (IBGE, 2001-2023)
_IPCA_ANUAL_2001_2023: Dict[int, float] = {
    2001: 0.0767, 2002: 0.1253, 2003: 0.0930, 2004: 0.0760, 2005: 0.0569,
    2006: 0.0314, 2007: 0.0446, 2008: 0.0590, 2009: 0.0431, 2010: 0.0591,
    2011: 0.0650, 2012: 0.0584, 2013: 0.0591, 2014: 0.0641, 2015: 0.1067,
    2016: 0.0629, 2017: 0.0295, 2018: 0.0375, 2019: 0.0431, 2020: 0.0452,
    2021: 0.1006, 2022: 0.0579, 2023: 0.0462,
}

# Contagens brutas em R$ constantes de dez/2023 (IPCA IBGE, cortes R$ 90 e R$ 260/MWh)
_T_BASE_2001_2023_IPCA = np.array([
    [25.0,  1.0,  9.0],  # de 0 (seco > 260 real)
    [ 0.0, 27.0,  6.0],  # de 1 (úmido < 90 real)
    [10.0,  5.0,  8.0],  # de 2 (moderado 90-260 real)
], dtype=float)


def _fator_ipca_para_dez2023(ano: int, mes: int) -> float:
    """Fator multiplicativo para trazer R$ correntes de (ano, mes) para R$ constantes de dez/2023."""
    if ano >= 2024:
        return 1.0
    f_fim_ano = 1.0
    for y in range(2023, ano, -1):
        f_fim_ano *= (1.0 + _IPCA_ANUAL_2001_2023[y])
    taxa_ano = _IPCA_ANUAL_2001_2023.get(ano, 0.0)
    frac_restante = (12.0 - float(mes) + 0.5) / 12.0
    return float(f_fim_ano * ((1.0 + taxa_ano) ** frac_restante))


def _classificar_regime_pld(pld_medio_trimestre: float, deflacionar_ipca: bool = False) -> int:
    """
    Classifica o trimestre hidrológico:
    - Nominal (deflacionar_ipca=False): 0=seco (>210), 1=úmido (<68), 2=normal (68-210).
    - Real dez/2023 (deflacionar_ipca=True): 0=seco (>260), 1=úmido (<90), 2=normal (90-260).
    """
    hi, lo = (260.0, 90.0) if deflacionar_ipca else (210.0, 68.0)
    if pld_medio_trimestre > hi:
        return 0
    if pld_medio_trimestre < lo:
        return 1
    return 2


def carregar_base_mensal_expansiva(
    ultimo_mes_realizado: Optional[int] = None,
    mensal_extra_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Carrega médias mensais de PLD_SE e PLD_NE de 2021-01 até `ultimo_mes_realizado` (formato YYYYMM).
    Quando `ultimo_mes_realizado` é None (ou <= 202312), retorna estritamente os 36 meses de treino (2021-2023).
    Quando `ultimo_mes_realizado >= 202401`, concatena os meses realizados de 2024-2025 (ou `mensal_extra_df`)
    até `ultimo_mes_realizado` inclusive, sem qualquer dado do mês de decisão em diante (sem look-ahead).
    """
    csv_treino = DATA_DIR / "pld_horario_treino_2021_2023.csv"
    if not csv_treino.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado em {csv_treino}")

    df_tr = pd.read_csv(csv_treino)
    mensal_tr = df_tr.groupby("MES_REFERENCIA")[["PLD_SE", "PLD_NE"]].mean()

    if ultimo_mes_realizado is None or int(ultimo_mes_realizado) <= 202312:
        if ultimo_mes_realizado is not None:
            mensal_tr = mensal_tr[mensal_tr.index <= int(ultimo_mes_realizado)]
        return mensal_tr.sort_index()

    if mensal_extra_df is not None:
        mensal_te = mensal_extra_df[mensal_extra_df.index <= int(ultimo_mes_realizado)][["PLD_SE", "PLD_NE"]]
        if len(mensal_te) == 0:
            return mensal_tr.sort_index()
        mensal_comb = pd.concat([mensal_tr, mensal_te]).sort_index()
        assert int(mensal_comb.index.max()) <= int(ultimo_mes_realizado)
        return mensal_comb

    csv_teste = DATA_DIR / "pld_horario_teste_2024_2025.csv"
    if not csv_teste.exists():
        raise FileNotFoundError(f"Arquivo de teste não encontrado em {csv_teste}")

    df_te = pd.read_csv(csv_teste)
    df_te_filt = df_te[df_te["MES_REFERENCIA"] <= int(ultimo_mes_realizado)]
    if len(df_te_filt) == 0:
        return mensal_tr.sort_index()

    mensal_te = df_te_filt.groupby("MES_REFERENCIA")[["PLD_SE", "PLD_NE"]].mean()
    mensal_comb = pd.concat([mensal_tr, mensal_te]).sort_index()
    assert int(mensal_comb.index.max()) <= int(ultimo_mes_realizado)
    return mensal_comb


def contar_blocos_por_celula(
    incluir_2001_2020: bool = False,
    ultimo_mes_realizado: Optional[int] = None,
    deflacionar_ipca: bool = False,
) -> Dict[int, Dict[int, int]]:
    """
    Conta blocos trimestrais distintos em cada célula (q_cal, regime) na calibração.
    Regimes: 0=seco, 1=úmido, 2=moderado (cortes nominais 68/210 ou reais IPCA 90/260).
    """
    counts: Dict[int, Dict[int, int]] = {q: {0: 0, 1: 0, 2: 0} for q in range(4)}

    if incluir_2001_2020:
        csv_01 = REPO_ROOT / "nyx" / "data" / "pld_mensal_oficial_se.csv"
        df_01 = pd.read_csv(csv_01)
        df_01_20 = df_01[(df_01["ano"] >= 2001) & (df_01["ano"] <= 2020)].sort_values(["ano", "mes"])
        if deflacionar_ipca:
            fatores = np.array(
                [_fator_ipca_para_dez2023(int(a), int(m)) for a, m in zip(df_01_20["ano"], df_01_20["mes"])],
                dtype=float,
            )
            vals_01_20 = np.clip(df_01_20["pld"].to_numpy(dtype=float) * fatores, 49.77, 684.73)
        else:
            vals_01_20 = np.clip(df_01_20["pld"].to_numpy(dtype=float), 49.77, 583.88)
        for i in range(len(vals_01_20) // 3):
            q_cal = i % 4
            reg = _classificar_regime_pld(float(np.mean(vals_01_20[i * 3:(i + 1) * 3])), deflacionar_ipca=deflacionar_ipca)
            counts[q_cal][reg] += 1

    mensal_df = carregar_base_mensal_expansiva(ultimo_mes_realizado)
    meses_ord = sorted(mensal_df.index)
    for i in range(len(meses_ord) // 3):
        q_cal = i % 4
        ms = meses_ord[i * 3:(i + 1) * 3]
        vals_q = mensal_df.loc[ms, "PLD_SE"].to_numpy(dtype=float)
        if deflacionar_ipca:
            fatores_q = np.array([_fator_ipca_para_dez2023(int(m) // 100, int(m) % 100) for m in ms], dtype=float)
            vals_q = np.clip(vals_q * fatores_q, 49.77, 684.73)
        reg = _classificar_regime_pld(float(np.mean(vals_q)), deflacionar_ipca=deflacionar_ipca)
        counts[q_cal][reg] += 1

    return counts


def construir_arvore_ccee_treino(
    divisao_estagios: Optional[List[List[int]]] = None,
    ramificacao: int = 2,
    seed: int = 2026,
    volatilidade_demanda: float = 0.08,
    correlacao_demanda_hidrologia: float = 0.0,
    trimestre_inicial: int = 0,
    ultimo_mes_realizado: Optional[int] = None,
    mensal_extra_df: Optional[pd.DataFrame] = None,
    n_amostras_mc: int = 1024,
    gerador_diversificado: bool = False,
    spread_ne_medio: float = 25.0,
    ramificar_conjunto: bool = True,
    deflacionar_ipca: bool = False,
) -> ArvoreCenarios:
    """
    Constrói árvore trimestral sobre n_months meses amostrados da base histórica da CCEE
    (2021-2023, ou janela expansiva até `ultimo_mes_realizado`).

    divisao_estagios: lista de meses por estágio (ex.: 8 trimestres [[0,1,2], ..., [21,22,23]],
                      ou horizonte encolhido [[0,1,2], ...]).
    trimestre_inicial: trimestre calendário do primeiro bloco (0=Q1, 1=Q2, 2=Q3, 3=Q4),
                       garantindo alinhamento sazonal em reotimizações rolantes.
    ultimo_mes_realizado: último mês calendário YYYYMM observado antes da decisão (ex.: 202403 para Q1).
                          Se None ou <= 202312, usa estritamente a base de treino 2021-2023.
    n_amostras_mc: número de trajetórias Monte Carlo simuladas antes da bisseção (padrão: 1024).
    gerador_diversificado: se True, amplia o pool de blocos com 2001-2020 (pld_mensal_oficial_se.csv)
                           + choque intra-bloco de 4% + modelo paramétrico de spread regional SE-NE.
    ramificar_conjunto: se True (padrão), usa bisseção pela 1a componente principal do vetor
                        padronizado conjunto (PLD_SE, PLD_NE, demanda_fator); se False, usa apenas PLD_SE.
    deflacionar_ipca: se True, traz os blocos históricos 2001-2023 para R$ constantes de dez/2023
                      via IPCA oficial (IBGE) e usa cortes reais (R$ 90 e R$ 260/MWh) e matriz IPCA.
    """
    if divisao_estagios is None:
        divisao_estagios = [list(range(q * 3, q * 3 + 3)) for q in range(8)]

    mensal_df = carregar_base_mensal_expansiva(ultimo_mes_realizado, mensal_extra_df=mensal_extra_df)
    meses_ord = sorted(mensal_df.index)
    modo_expansivo = ultimo_mes_realizado is not None and int(ultimo_mes_realizado) >= 202401

    # Organizar os blocos trimestrais observados
    blocos_se: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
    blocos_ne: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
    blocos_se_reg: Dict[int, Dict[int, List[np.ndarray]]] = {q: {0: [], 1: [], 2: []} for q in range(4)}
    blocos_ne_reg: Dict[int, Dict[int, List[np.ndarray]]] = {q: {0: [], 1: [], 2: []} for q in range(4)}
    blocos_se_global: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    blocos_ne_global: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    regimes_trimestres: List[int] = []

    if gerador_diversificado:
        csv_01 = REPO_ROOT / "nyx" / "data" / "pld_mensal_oficial_se.csv"
        df_01 = pd.read_csv(csv_01)
        df_01_20 = df_01[(df_01["ano"] >= 2001) & (df_01["ano"] <= 2020)].sort_values(["ano", "mes"])
        if deflacionar_ipca:
            fatores = np.array(
                [_fator_ipca_para_dez2023(int(a), int(m)) for a, m in zip(df_01_20["ano"], df_01_20["mes"])],
                dtype=float,
            )
            vals_01_20 = np.clip(df_01_20["pld"].to_numpy(dtype=float) * fatores, 49.77, 684.73)
        else:
            vals_01_20 = np.clip(df_01_20["pld"].to_numpy(dtype=float), 49.77, 583.88)
        for i in range(len(vals_01_20) // 3):
            q_cal = i % 4
            v_se = vals_01_20[i * 3:(i + 1) * 3].copy()
            reg = _classificar_regime_pld(float(np.mean(v_se)), deflacionar_ipca=deflacionar_ipca)
            blocos_se[q_cal].append(v_se)
            blocos_se_reg[q_cal][reg].append(v_se)
            blocos_se_global[reg].append(v_se)

    n_q_obs = len(meses_ord) // 3
    for i in range(n_q_obs):
        q_cal = i % 4
        ms = meses_ord[i * 3:(i + 1) * 3]
        v_se = mensal_df.loc[ms, "PLD_SE"].to_numpy(dtype=float)
        v_ne = mensal_df.loc[ms, "PLD_NE"].to_numpy(dtype=float)
        if deflacionar_ipca:
            fatores_q = np.array([_fator_ipca_para_dez2023(int(m) // 100, int(m) % 100) for m in ms], dtype=float)
            v_se = np.clip(v_se * fatores_q, 49.77, 684.73)
            v_ne = np.clip(v_ne * fatores_q, 49.77, 684.73)
        reg = _classificar_regime_pld(float(np.mean(v_se)), deflacionar_ipca=deflacionar_ipca)
        regimes_trimestres.append(reg)

        blocos_se[q_cal].append(v_se)
        blocos_ne[q_cal].append(v_ne)
        blocos_se_reg[q_cal][reg].append(v_se)
        blocos_ne_reg[q_cal][reg].append(v_ne)
        blocos_se_global[reg].append(v_se)
        blocos_ne_global[reg].append(v_ne)

    rng = np.random.RandomState(seed)
    n_months = max(max(ms) for ms in divisao_estagios if ms) + 1
    n_stages = len(divisao_estagios)
    n_q_sim = (n_months + 2) // 3
    N = int(n_amostras_mc)  # Amostras para bootstrap em blocos

    # Matriz de transição: base 2001-2023 (91 transições) + transições realizadas de 2024+ no modo expansivo
    T_mat = (_T_BASE_2001_2023_IPCA if deflacionar_ipca else _T_BASE_2001_2023).copy()
    if modo_expansivo and n_q_obs > 12:
        # Índice 11 = 2023Q4; somar apenas transições cujo destino é >= 2024Q1 (índice >= 12)
        for idx_q in range(12, n_q_obs):
            r_prev = regimes_trimestres[idx_q - 1]
            r_curr = regimes_trimestres[idx_q]
            T_mat[r_prev, r_curr] += 1.0

    P_trans = (T_mat + 1.0) / (T_mat.sum(axis=1, keepdims=True) + 3.0)

    trajetorias_se = np.zeros((N, n_months), dtype=float)
    trajetorias_ne = np.zeros((N, n_months), dtype=float)
    trajetorias_dem = np.zeros((N, n_months), dtype=float)

    # Distribuição empírica inicial para Q0 / in-sample (2001-2023)
    p_inicial = [0.38, 0.36, 0.26] if deflacionar_ipca else [0.25, 0.40, 0.35]
    estado_condicionado = regimes_trimestres[-1] if (modo_expansivo and regimes_trimestres) else None

    for i in range(N):
        estado = estado_condicionado if estado_condicionado is not None else rng.choice([0, 1, 2], p=p_inicial)
        dem_acum = 1.0
        for q in range(n_q_sim):
            q_cal = (trimestre_inicial + q) % 4
            estado = rng.choice([0, 1, 2], p=P_trans[estado])

            if gerador_diversificado:
                pool_se = blocos_se_reg[q_cal][estado]
                if len(pool_se) == 0:
                    pool_se = blocos_se_global[estado]
                if len(pool_se) == 0:
                    pool_se = blocos_se[q_cal]
                idx_b = int(rng.randint(len(pool_se)))
                ruido_se = rng.normal(1.0, 0.04, size=3)
                b_se = np.clip(pool_se[idx_b] * ruido_se, 49.77, 751.73)
                sp_q = (
                    rng.exponential(spread_ne_medio, size=3)
                    if rng.rand() < 0.30
                    else rng.exponential(4.0, size=3)
                )
                b_ne = np.clip(b_se - sp_q + rng.normal(0.0, 3.0, size=3), 49.77, 751.73)
            elif not modo_expansivo:
                # Ruído empírico moderado intra-bloco (~2%)
                ruido = rng.normal(1.0, 0.02)
                b_se = blocos_se[q_cal][estado] * ruido
                b_ne = blocos_ne[q_cal][estado] * ruido
            else:
                ruido = rng.normal(1.0, 0.02)
                pool_se = blocos_se_reg[q_cal][estado]
                pool_ne = blocos_ne_reg[q_cal][estado]
                if len(pool_se) == 0:
                    pool_se = blocos_se_global[estado]
                    pool_ne = blocos_ne_global[estado]
                if len(pool_se) == 0:
                    pool_se = blocos_se[q_cal]
                    pool_ne = blocos_ne[q_cal]
                idx_b = int(rng.randint(len(pool_se)))
                b_se = pool_se[idx_b] * ruido
                b_ne = pool_ne[idx_b] * ruido

            m_start = q * 3
            m_end = min(m_start + 3, n_months)
            n_slice = m_end - m_start
            trajetorias_se[i, m_start:m_end] = b_se[:n_slice]
            trajetorias_ne[i, m_start:m_end] = b_ne[:n_slice]

            # Incerteza de demanda
            if correlacao_demanda_hidrologia > 0.0:
                sinal = -1.0 if estado == 0 else (1.0 if estado == 1 else 0.0)
                mu_c = correlacao_demanda_hidrologia * sinal * volatilidade_demanda
                sig_c = np.sqrt(max(0.0, 1.0 - correlacao_demanda_hidrologia**2)) * volatilidade_demanda
                choque_d = rng.normal(mu_c, max(0.005, sig_c))
            else:
                choque_d = rng.normal(0.0, volatilidade_demanda)

            lim_inf = max(0.40, 1.0 - 2.0 * max(0.10, volatilidade_demanda))
            lim_sup = min(1.60, 1.0 + 2.0 * max(0.10, volatilidade_demanda))
            dem_acum = np.clip(dem_acum * (1.0 + choque_d), lim_inf, lim_sup)
            trajetorias_dem[i, m_start:m_end] = dem_acum

    # Desvios-padrão globais na raiz para padronização conjunta sem viés de escala
    global_std_se = np.maximum(trajetorias_se.std(axis=0), 1e-9)
    global_std_ne = np.maximum(trajetorias_ne.std(axis=0), 1e-9)
    global_std_dem = np.maximum(trajetorias_dem.std(axis=0), 1e-9)

    def _dividir_cluster_kmeans(indices: List[int], start_m: int) -> List[List[int]]:
        """
        Divide índices de trajetórias em 2 subclusters usando projeção no primeiro componente principal:
        - Se ramificar_conjunto=True: vetor conjunto padronizado (PLD_SE, PLD_NE, demanda_fator).
        - Se ramificar_conjunto=False: apenas PLD_SE.
        """
        if len(indices) <= 1:
            return [indices]
        if ramificar_conjunto:
            s_se = global_std_se[start_m:]
            s_ne = global_std_ne[start_m:]
            s_dm = global_std_dem[start_m:]
            Z_se = (trajetorias_se[indices, start_m:] - trajetorias_se[indices, start_m:].mean(axis=0)) / s_se
            Z_ne = (trajetorias_ne[indices, start_m:] - trajetorias_ne[indices, start_m:].mean(axis=0)) / s_ne
            Z_dm = (trajetorias_dem[indices, start_m:] - trajetorias_dem[indices, start_m:].mean(axis=0)) / s_dm
            X_cent = np.hstack([Z_se, Z_ne, Z_dm])
            col_vars = np.var(X_cent, axis=0)
            v = np.zeros(X_cent.shape[1], dtype=float)
            v[int(np.argmax(col_vars))] = 1.0
            n_iter = 15
        else:
            X_sub = trajetorias_se[indices, start_m:]
            X_cent = X_sub - X_sub.mean(axis=0)
            v = np.ones(X_sub.shape[1], dtype=float)
            n_iter = 5
        for _ in range(n_iter):
            v = X_cent.T @ (X_cent @ v)
            norm_v = np.linalg.norm(v)
            if norm_v > 0:
                v /= norm_v
        scores = X_cent @ v
        med = np.median(scores)
        c0 = [indices[k] for k in range(len(indices)) if scores[k] >= med]
        c1 = [indices[k] for k in range(len(indices)) if scores[k] < med]
        if len(c0) == 0 or len(c1) == 0:
            c0 = indices[:len(indices) // 2]
            c1 = indices[len(indices) // 2:]
        return [c0, c1]

    # Construção da árvore de cenários
    nodes: Dict[int, NoCenario] = {}
    stages: List[List[int]] = [[] for _ in range(n_stages)]
    node_indices_map: Dict[int, List[int]] = {}

    raiz_indices = list(range(N))
    raiz = NoCenario(
        id=0,
        parent_id=None,
        stage=0,
        meses=divisao_estagios[0],
        prob=1.0,
        cond_prob=1.0,
        cenarios_indices=raiz_indices,
        pld=trajetorias_se.mean(axis=0),
        pld_se=trajetorias_se.mean(axis=0),
        pld_ne=trajetorias_ne.mean(axis=0),
        demanda_fator=trajetorias_dem.mean(axis=0),
    )
    nodes[0] = raiz
    stages[0].append(0)
    node_indices_map[0] = raiz_indices
    counter = 1

    for s_idx in range(1, n_stages):
        meses_estagio = divisao_estagios[s_idx]
        start_m = meses_estagio[0]
        for p_id in stages[s_idx - 1]:
            p_indices = node_indices_map[p_id]
            subclusters = _dividir_cluster_kmeans(p_indices, start_m)
            for c in subclusters:
                nid = counter
                counter += 1
                cond_p = len(c) / len(p_indices)
                prob = nodes[p_id].prob * cond_p
                pld_se_node = trajetorias_se[c].mean(axis=0)
                pld_ne_node = trajetorias_ne[c].mean(axis=0)
                dem_node = trajetorias_dem[c].mean(axis=0)

                node = NoCenario(
                    id=nid,
                    parent_id=p_id,
                    stage=s_idx,
                    meses=meses_estagio,
                    prob=prob,
                    cond_prob=cond_p,
                    cenarios_indices=c,
                    pld=pld_se_node.copy(),
                    pld_se=pld_se_node.copy(),
                    pld_ne=pld_ne_node.copy(),
                    demanda_fator=dem_node.copy(),
                )
                nodes[nid] = node
                stages[s_idx].append(nid)
                nodes[p_id].children_ids.append(nid)
                node_indices_map[nid] = c

    return ArvoreCenarios(
        nodes=nodes,
        stages=stages,
        n_months=n_months,
        n_series_originais=len(stages[-1]),
        divisao_estagios=divisao_estagios,
    )


def calcular_dispersao_folhas(arvore: ArvoreCenarios) -> Dict[str, float]:
    """Calcula estatísticas de dispersão entre folhas para theta (demanda), PLD_SE e PLD_NE (meses m >= 3)."""
    leaves = arvore.leaves
    dems = np.array([l.demanda_fator for l in leaves], dtype=float)
    se = np.array([l.pld_se for l in leaves], dtype=float)
    ne = np.array([l.pld_ne for l in leaves], dtype=float)
    sl = slice(3, arvore.n_months) if arvore.n_months > 3 else slice(0, arvore.n_months)
    d_sub = dems[:, sl]
    se_sub = se[:, sl]
    ne_sub = ne[:, sl]
    return {
        "n_folhas": float(len(leaves)),
        "theta_sd_medio": float(d_sub.std(axis=0).mean()),
        "theta_iqr_medio": float((np.percentile(d_sub, 75, axis=0) - np.percentile(d_sub, 25, axis=0)).mean()),
        "theta_min": float(d_sub.min()),
        "theta_max": float(d_sub.max()),
        "pld_se_sd_medio": float(se_sub.std(axis=0).mean()),
        "pld_ne_sd_medio": float(ne_sub.std(axis=0).mean()),
    }

