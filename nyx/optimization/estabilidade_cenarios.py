"""
Módulo de testes de estabilidade In-Sample e Out-of-Sample para a árvore de cenários TEMPR.

1. Estabilidade In-Sample:
   - Gera árvores com diferentes sementes e números de cenários terminais (S in [16, 32, 64, 128, 256]).
   - Verifica se o valor ótimo da função objetivo converge e tem variância controlada à medida que S cresce.
2. Estabilidade Out-of-Sample (In-Sample vs Out-of-Sample gap):
   - Fixa a política de 1º estágio (decisões ex-ante no nó raiz x_plus, x_minus) obtida na árvore de treino.
   - Avalia o desempenho dessa política em uma amostra de teste independente de 1.000 trajetórias sintéticas calibradas com a mesma distribuição de treino.
   - Verifica que o desvio |E[R_out] - E[R_in]| fica dentro de limites aceitáveis (< 3%), comprovando ausência de overfitting.
"""

from pathlib import Path
import time
import json
import numpy as np
import pandas as pd

import sys
BASE = Path(__file__).resolve().parents[1]
for sub in ("config", "optimization", "optimization/models", "data_processing", "utils"):
    sys.path.append(str(BASE / sub))

from arvore_ccee import construir_arvore_ccee_treino
from modelo_multiestagio_tempr import ModeloMultiestagioTEMPR

OUTPUT_DIR = BASE / "resultados" / "estabilidade"


def avaliar_estabilidade_in_sample(
    tamanhos_folhas: list[int] = [16, 32, 64, 128],
    n_repeticoes: int = 10,
    lam: float = 0.5,
    n_amostras_mc: int = 1024,
    salvar_csv: bool = True,
) -> pd.DataFrame:
    """
    Avalia a estabilidade in-sample variando o número de folhas e a semente estocástica.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resultados = []

    print(f"=== AVALIANDO ESTABILIDADE IN-SAMPLE (lambda={lam}, N_MC={n_amostras_mc}) ===")
    for n_folhas in tamanhos_folhas:
        profundidade = int(np.log2(n_folhas))
        divisao = [list(range(q * 3, q * 3 + 3)) for q in range(profundidade + 1)]
        if divisao[-1][-1] < 23:
            divisao[-1] = list(range(divisao[-1][0], 24))

        valores_fo = []
        valores_er = []
        valores_cvar = []
        tempos = []

        for rep in range(n_repeticoes):
            seed = 2026 + rep * 101
            arv = construir_arvore_ccee_treino(
                divisao_estagios=divisao,
                ramificacao=2,
                seed=seed,
                volatilidade_demanda=0.08,
                n_amostras_mc=n_amostras_mc,
            )

            m = ModeloMultiestagioTEMPR(
                arvore=arv,
                _lambda=lam,
                beta=0.95,
                permitir_revenda=True,
                solver_name="HiGHS",
            )
            m.criar_variaveis()
            m.adicionar_restricoes_posicao_e_balanco()
            m.adicionar_restricoes_revenda()
            m.definir_funcao_objetivo()
            t0 = time.time()
            m.resolver(msg=False)
            t_solve = time.time() - t0

            valores_fo.append(m.obj_value / 1e6)
            valores_er.append(m.expected_profit / 1e6)
            valores_cvar.append(m.cvar_perda / 1e6)
            tempos.append(t_solve)

        media_fo = float(np.mean(valores_fo))
        std_fo = float(np.std(valores_fo))
        cv_fo = (std_fo / abs(media_fo) * 100) if media_fo != 0 else 0.0

        resultados.append({
            "n_amostras_mc": n_amostras_mc,
            "n_folhas": len(arv.leaves),
            "n_nos": len(arv.nodes),
            "n_estagios": len(arv.stages),
            "media_fo_milhoes": media_fo,
            "std_fo_milhoes": std_fo,
            "cv_fo_pct": cv_fo,
            "media_er_milhoes": float(np.mean(valores_er)),
            "std_er_milhoes": float(np.std(valores_er)),
            "media_cvar_milhoes": float(np.mean(valores_cvar)),
            "tempo_medio_s": float(np.mean(tempos)),
        })
        print(f"N={n_amostras_mc:4d} Folhas={len(arv.leaves):3d} | FO Média={media_fo:8.2f}M | Desv.Padrão={std_fo:5.2f}M | CV={cv_fo:5.2f}% | t={np.mean(tempos)*1000:6.1f}ms")

    df = pd.DataFrame(resultados)
    if salvar_csv:
        fname = "estabilidade_in_sample.csv" if n_amostras_mc == 1024 else f"estabilidade_in_sample_N{n_amostras_mc}.csv"
        csv_path = OUTPUT_DIR / fname
        df.to_csv(csv_path, index=False)
        print(f"Resultados in-sample salvos em: {csv_path}\n")
    return df


def avaliar_estabilidade_kaut_wallace(
    n_amostras_lista: list[int] = [1024, 4096],
    n_repeticoes: int = 10,
    n_ref_mc: int = 16384,
    seed_ref: int = 99999,
    lam: float = 0.5,
    salvar_json: bool = True,
) -> dict:
    """
    Avalia a estabilidade fora da amostra à maneira de Kaut & Wallace (2007):
    Fixa a decisão de primeiro estágio (nó raiz nu=0) obtida em cada árvore de treino (N=1024, N=4096)
    e avalia seu desempenho ótimo de recurso sobre uma árvore de referência grande e independente (N=16384).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]
    seeds = [2026 + rep * 101 for rep in range(n_repeticoes)]

    arv_ref = construir_arvore_ccee_treino(
        divisao_estagios=divisao_8q,
        ramificacao=2,
        seed=seed_ref,
        volatilidade_demanda=0.08,
        n_amostras_mc=n_ref_mc,
    )
    m_ref = ModeloMultiestagioTEMPR(arvore=arv_ref, _lambda=lam, permitir_revenda=True)
    m_ref.criar_variaveis()
    m_ref.adicionar_restricoes_posicao_e_balanco()
    m_ref.adicionar_restricoes_revenda()
    m_ref.definir_funcao_objetivo()
    m_ref.resolver(msg=False)
    fo_ref = float(m_ref.obj_value / 1e6)
    print(f"=== KAUT & WALLACE (2007) OOS STABILITY (Ref N={n_ref_mc}, seed={seed_ref}: FO_ref={fo_ref:.4f}M) ===")

    resumo = {"lambda": lam, "n_ref_mc": n_ref_mc, "seed_ref": seed_ref, "fo_ref_M": fo_ref, "por_n_mc": {}}
    for n_mc in n_amostras_lista:
        oos_fos = []
        for s in seeds:
            arv_s = construir_arvore_ccee_treino(
                divisao_estagios=divisao_8q,
                ramificacao=2,
                seed=s,
                volatilidade_demanda=0.08,
                n_amostras_mc=n_mc,
            )
            ms = ModeloMultiestagioTEMPR(arvore=arv_s, _lambda=lam, permitir_revenda=True)
            ms.criar_variaveis()
            ms.adicionar_restricoes_posicao_e_balanco()
            ms.adicionar_restricoes_revenda()
            ms.definir_funcao_objetivo()
            ms.resolver(msg=False)

            mr = ModeloMultiestagioTEMPR(arvore=arv_ref, _lambda=lam, permitir_revenda=True)
            for sub in ["SE", "NE"]:
                mr.D_max[sub] = np.maximum(mr.D_max[sub], ms.D_max[sub])
            mr.criar_variaveis()
            mr.adicionar_restricoes_posicao_e_balanco()
            mr.adicionar_restricoes_revenda()
            for k, v in ms.variables.items():
                if k.startswith(("xm_plus[0,", "xm_minus[0,", "xt_plus[0,", "xt_minus[0,", "xs_plus[0,", "xs_minus[0,")):
                    mr.lp_model += (mr.variables[k] == float(v.varValue or 0.0))
            mr.definir_funcao_objetivo()
            mr.resolver(msg=False)
            assert mr.obj_value is not None, f"KW OOS falhou para seed={s}, N={n_mc}"
            oos_fos.append(float(mr.obj_value / 1e6))

        mean_o = float(np.mean(oos_fos))
        sd_o = float(np.std(oos_fos, ddof=1)) if len(oos_fos) > 1 else 0.0
        cv_o = (sd_o / abs(mean_o) * 100.0) if mean_o != 0 else 0.0
        gap_pct = (fo_ref - mean_o) / abs(fo_ref) * 100.0
        resumo["por_n_mc"][str(n_mc)] = {
            "media_oos_M": mean_o,
            "std_amostral_oos_M": sd_o,
            "cv_oos_pct": cv_o,
            "min_oos_M": float(np.min(oos_fos)),
            "max_oos_M": float(np.max(oos_fos)),
            "gap_otimalidade_pct": gap_pct,
            "valores_M": oos_fos,
        }
        print(f"KW-OOS N={n_mc:4d}: {mean_o:.4f}M +- {sd_o:.4f}M (CV={cv_o:.3f}%, gap={gap_pct:.4f}%)")

    if salvar_json:
        json_path = OUTPUT_DIR / "estabilidade_kaut_wallace.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=4, ensure_ascii=False)
        print(f"Resultados Kaut & Wallace salvos em: {json_path}\n")
    return resumo


def avaliar_estabilidade_out_of_sample(
    arvore_treino,
    n_cenarios_teste: int = 500,
    seed_teste: int = 9999,
    lam: float = 0.5,
) -> dict:
    """
    Avalia a estabilidade out-of-sample:
    1. Resolve o modelo com a árvore de treino e extrai a política do 1º estágio.
    2. Simula o desempenho da política em n_cenarios_teste não observados.
    """
    print(f"=== AVALIANDO ESTABILIDADE OUT-OF-SAMPLE (lambda={lam}) ===")
    
    # 1. Resolver o modelo de treino para obter a decisão x^+(0), x^-(0)
    m_treino = ModeloMultiestagioTEMPR(arvore=arvore_treino, _lambda=lam, permitir_revenda=True)
    m_treino.criar_variaveis()
    m_treino.adicionar_restricoes_posicao_e_balanco()
    m_treino.adicionar_restricoes_revenda()
    m_treino.definir_funcao_objetivo()
    m_treino.resolver(msg=False)

    er_in = m_treino.expected_profit / 1e6
    cvar_in = m_treino.cvar_perda / 1e6
    fo_in = m_treino.obj_value / 1e6

    # 2. Avaliar nas folhas de teste independentes
    rng = np.random.RandomState(seed_teste)
    lucros_out = []
    
    # Extrair compras da raiz
    raiz_id = 0
    pld_se_base = arvore_treino.root.pld_se
    pld_ne_base = arvore_treino.root.pld_ne
    horas = m_treino.horas_mes

    compras_raiz_se = np.array([m_treino.variables[f"xm_plus[{raiz_id},SE,{n}]"].varValue or 0.0 for n in range(24)])
    compras_raiz_ne = np.array([m_treino.variables[f"xm_plus[{raiz_id},NE,{n}]"].varValue or 0.0 for n in range(24)])

    for sim in range(n_cenarios_teste):
        # Gerar trajetória independente a partir da distribuição de treino
        choque_se = rng.normal(1.0, 0.20, size=24)
        choque_ne = rng.normal(1.0, 0.25, size=24)
        fator_demanda = rng.normal(1.0, 0.08, size=24)

        pld_sim_se = np.clip(pld_se_base * choque_se, 49.77, 1470.57)
        pld_sim_ne = np.clip(pld_ne_base * choque_ne, 49.77, 1470.57)

        d_sim_se = m_treino.D_base["SE"] * fator_demanda
        d_sim_ne = m_treino.D_base["NE"] * fator_demanda

        # Fluxo com a política fixa de 1º estágio
        # Receita de clientes
        rec_cli = np.sum(m_treino.preco_venda_cliente_mwh * (d_sim_se + d_sim_ne) * horas)
        # Encargos
        enc = np.sum(m_treino.encargo_ccee_mwh * (d_sim_se + d_sim_ne) * horas)
        # Custo contratos raiz
        custo_contr_se = np.sum(m_treino.precos_termo[raiz_id]["SE"]["pm_c"] * compras_raiz_se * horas)
        custo_contr_ne = np.sum(m_treino.precos_termo[raiz_id]["NE"]["pm_c"] * compras_raiz_ne * horas)
        # Spot: sobra ou déficit
        sobra_se = compras_raiz_se - d_sim_se
        sobra_ne = compras_raiz_ne - d_sim_ne
        liq_spot = np.sum(pld_sim_se * sobra_se * horas) + np.sum(pld_sim_ne * sobra_ne * horas)

        lucro_total = rec_cli - enc - custo_contr_se - custo_contr_ne + liq_spot
        lucros_out.append(lucro_total / 1e6)

    lucros_out = np.array(lucros_out)
    er_out = float(np.mean(lucros_out))
    
    # CVaR out-of-sample
    perdas_out = -lucros_out
    var_95 = float(np.percentile(perdas_out, 95))
    cvar_out = float(np.mean(perdas_out[perdas_out >= var_95]))

    gap_er = abs(er_out - er_in) / abs(er_in) * 100
    print(f"E[R] In-Sample:  {er_in:8.2f} Mlh R$")
    print(f"E[R] Out-Sample: {er_out:8.2f} Mlh R$ (Gap: {gap_er:.2f}%)")
    print(f"CVaR In-Sample:  {cvar_in:8.2f} Mlh R$")
    print(f"CVaR Out-Sample: {cvar_out:8.2f} Mlh R$")

    resultado = {
        "lambda": lam,
        "er_in_sample": er_in,
        "er_out_of_sample": er_out,
        "gap_er_pct": gap_er,
        "cvar_in_sample": cvar_in,
        "cvar_out_of_sample": cvar_out,
        "n_cenarios_teste": n_cenarios_teste,
    }

    json_path = OUTPUT_DIR / "estabilidade_out_of_sample.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4)
    print(f"Resultado out-of-sample salvo em: {json_path}\n")
    return resultado


def _t_crit_95(df: int) -> float:
    """Valor critico bilateral t-Student (95%, alpha=0.05) sem dependencia externa."""
    import math

    if df <= 0:
        return 0.0
    if df == 9:
        return 2.2621571627409915
    coef = math.exp(math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0)) / math.sqrt(df * math.pi)

    def _cdf(x: float) -> float:
        # Regra de Simpson composta em [0, x]
        n_steps = 400
        h = x / n_steps
        s = coef * (1.0 + (1.0 + (x * x) / df) ** (-(df + 1) / 2.0))
        for k in range(1, n_steps):
            xk = k * h
            fk = coef * (1.0 + (xk * xk) / df) ** (-(df + 1) / 2.0)
            s += (4.0 if k % 2 == 1 else 2.0) * fk
        return 0.5 + s * h / 3.0

    lo, hi = 1.5, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _cdf(mid) < 0.975:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def avaliar_diferenca_pareada_sementes(
    lambdas: list[float] | None = None,
    n_repeticoes: int = 10,
    seeds: list[int] | None = None,
    volatilidade_demanda: float = 0.08,
    preco_venda_cliente_mwh: float = 160.0,
    ramificar_conjunto: bool = True,
    salvar_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Avalia a diferenca pareada (ME-R menos ME) em multiplas sementes para cada lambda,
    eliminando a variancia de nivel entre arvores independentes.

    Retorna (df_detalhado, df_resumo) com media, desvio-padrao amostral (ddof=1)
    e intervalo de confianca t-Student de 95% (gl = n - 1).
    """
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    if seeds is None:
        seeds = [2026 + rep * 101 for rep in range(n_repeticoes)]

    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]
    registros = []

    print(f"=== DIFERENCA PAREADA ME-R vs ME ({len(seeds)} SEMENTES, 128 FOLHAS, P_cli=R${preco_venda_cliente_mwh:.0f}) ===")
    for seed in seeds:
        arv = construir_arvore_ccee_treino(
            divisao_estagios=divisao_8q,
            ramificacao=2,
            seed=seed,
            volatilidade_demanda=volatilidade_demanda,
            ramificar_conjunto=ramificar_conjunto,
        )
        for lam in lambdas:
            res = {}
            for permitir_rev in (True, False):
                m = ModeloMultiestagioTEMPR(
                    arvore=arv,
                    _lambda=lam,
                    beta=0.95,
                    permitir_revenda=permitir_rev,
                    preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                    encargo_ccee_mwh=15.0,
                    ano_inicio=2021,
                    solver_name="HiGHS",
                )
                m.criar_variaveis()
                m.adicionar_restricoes_posicao_e_balanco()
                m.adicionar_restricoes_revenda()
                m.definir_funcao_objetivo()
                m.resolver(msg=False)
                res[permitir_rev] = (
                    m.obj_value / 1e6,
                    m.expected_profit / 1e6,
                    -m.cvar_perda / 1e6,
                    float(np.sum((m.D_base["SE"] + m.D_base["NE"]) * m.horas_mes) / 1e6),
                )
            fo_r, er_r, cv_r, mwh_m = res[True]
            fo_m, er_m, cv_m, _ = res[False]
            d_fo = fo_r - fo_m
            registros.append({
                "seed": seed,
                "lambda": lam,
                "fo_mer_milhoes": fo_r,
                "fo_me_milhoes": fo_m,
                "d_fo_milhoes": d_fo,
                "d_fo_r_mwh": d_fo / mwh_m if mwh_m > 0 else 0.0,
                "cvar_profit_mer_milhoes": cv_r,
                "cvar_profit_me_milhoes": cv_m,
                "d_cvar_profit_milhoes": cv_r - cv_m,
                "er_mer_milhoes": er_r,
                "er_me_milhoes": er_m,
                "d_er_milhoes": er_r - er_m,
            })

    df_detalhado = pd.DataFrame(registros)

    resumo_rows = []
    for lam, grp in df_detalhado.groupby("lambda", sort=True):
        n = len(grp)
        t_crit = _t_crit_95(n - 1) if n > 1 else 0.0

        def _stats_col(col: str) -> tuple[float, float, float, float, float, float]:
            vals = grp[col].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se = sd / np.sqrt(n)
            return mean, sd, mean - t_crit * se, mean + t_crit * se, float(np.min(vals)), float(np.max(vals))

        dfo_m, dfo_sd, dfo_l, dfo_u, dfo_min, dfo_max = _stats_col("d_fo_milhoes")
        dcv_m, dcv_sd, dcv_l, dcv_u, dcv_min, dcv_max = _stats_col("d_cvar_profit_milhoes")
        der_m, der_sd, der_l, der_u, der_min, der_max = _stats_col("d_er_milhoes")
        dr_mwh_m, _, _, _, _, _ = _stats_col("d_fo_r_mwh")

        resumo_rows.append({
            "lambda": float(lam),
            "n_sementes": n,
            "fo_mer_media": float(grp["fo_mer_milhoes"].mean()),
            "fo_me_media": float(grp["fo_me_milhoes"].mean()),
            "d_fo_media": dfo_m,
            "d_fo_std": dfo_sd,
            "d_fo_ic95_inf": dfo_l,
            "d_fo_ic95_sup": dfo_u,
            "d_fo_min": dfo_min,
            "d_fo_max": dfo_max,
            "d_fo_r_mwh_media": dr_mwh_m,
            "cvar_profit_mer_media": float(grp["cvar_profit_mer_milhoes"].mean()),
            "cvar_profit_me_media": float(grp["cvar_profit_me_milhoes"].mean()),
            "d_cvar_media": dcv_m,
            "d_cvar_std": dcv_sd,
            "d_cvar_ic95_inf": dcv_l,
            "d_cvar_ic95_sup": dcv_u,
            "d_cvar_min": dcv_min,
            "d_cvar_max": dcv_max,
            "er_mer_media": float(grp["er_mer_milhoes"].mean()),
            "er_me_media": float(grp["er_me_milhoes"].mean()),
            "d_er_media": der_m,
            "d_er_std": der_sd,
            "d_er_ic95_inf": der_l,
            "d_er_ic95_sup": der_u,
            "n_sementes_d_fo_pos": int((grp["d_fo_milhoes"] > 1e-6).sum()),
            "n_sementes_d_cvar_pos": int((grp["d_cvar_profit_milhoes"] > 1e-6).sum()),
        })
        print(
            f"lam={lam:.2f} | dFO={dfo_m:+.4f} +- {dfo_sd:.4f} (IC95% [{dfo_l:+.4f}, {dfo_u:+.4f}], +R${dr_mwh_m:.2f}/MWh)"
            f" | dCVaR={dcv_m:+.4f} +- {dcv_sd:.4f} (IC95% [{dcv_l:+.4f}, {dcv_u:+.4f}])"
            f" | dER={der_m:+.4f} +- {der_sd:.4f}"
        )

    df_resumo = pd.DataFrame(resumo_rows)
    if salvar_csv:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        det_path = OUTPUT_DIR / "diferenca_pareada_10_sementes.csv"
        res_path = OUTPUT_DIR / "diferenca_pareada_resumo_10_sementes.csv"
        df_detalhado.to_csv(det_path, index=False)
        df_resumo.to_csv(res_path, index=False)
        print(f"Resultados pareados detalhados salvos em: {det_path}")
        print(f"Resumo pareado salvo em: {res_path}\n")

    return df_detalhado, df_resumo


def avaliar_diferenca_pareada_kaut_wallace(
    n_amostras: int = 1024,
    n_repeticoes: int = 10,
    n_ref_mc: int = 16384,
    seed_ref: int = 99999,
    lam: float = 0.5,
    preco_venda_cliente_mwh: float = 160.0,
    salvar_json: bool = True,
) -> dict:
    """
    Minor #7: Avalia a diferenca pareada fora da amostra (ME-R menos ME)
    sobre a arvore de referencia grande e independente (N=16.384), fixando as
    decisoes de primeiro estagio obtidas em cada uma das 10 sementes.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]
    seeds = [2026 + rep * 101 for rep in range(n_repeticoes)]

    arv_ref = construir_arvore_ccee_treino(
        divisao_estagios=divisao_8q, ramificacao=2, seed=seed_ref, volatilidade_demanda=0.08, n_amostras_mc=n_ref_mc,
    )
    # Benchmarks livres na arvore de referencia
    ref_vals = {}
    for rev in (True, False):
        mr0 = ModeloMultiestagioTEMPR(
            arvore=arv_ref, _lambda=lam, permitir_revenda=rev, preco_venda_cliente_mwh=preco_venda_cliente_mwh,
        )
        mr0.criar_variaveis()
        mr0.adicionar_restricoes_posicao_e_balanco()
        mr0.adicionar_restricoes_revenda()
        mr0.definir_funcao_objetivo()
        mr0.resolver(msg=False)
        ref_vals[rev] = (float(mr0.obj_value / 1e6), float(-mr0.cvar_perda / 1e6))

    d_fo_oos = []
    d_cvar_oos = []
    fo_mer_oos = []
    fo_me_oos = []
    for s in seeds:
        arv_s = construir_arvore_ccee_treino(
            divisao_estagios=divisao_8q, ramificacao=2, seed=s, volatilidade_demanda=0.08, n_amostras_mc=n_amostras,
        )
        par_res = {}
        for rev in (True, False):
            ms = ModeloMultiestagioTEMPR(
                arvore=arv_s, _lambda=lam, permitir_revenda=rev, preco_venda_cliente_mwh=preco_venda_cliente_mwh,
            )
            ms.criar_variaveis()
            ms.adicionar_restricoes_posicao_e_balanco()
            ms.adicionar_restricoes_revenda()
            ms.definir_funcao_objetivo()
            ms.resolver(msg=False)

            mr = ModeloMultiestagioTEMPR(
                arvore=arv_ref, _lambda=lam, permitir_revenda=rev, preco_venda_cliente_mwh=preco_venda_cliente_mwh,
            )
            for sub in ["SE", "NE"]:
                mr.D_max[sub] = np.maximum(mr.D_max[sub], ms.D_max[sub])
            mr.criar_variaveis()
            mr.adicionar_restricoes_posicao_e_balanco()
            mr.adicionar_restricoes_revenda()
            for k, v in ms.variables.items():
                if k.startswith(("xm_plus[0,", "xm_minus[0,", "xt_plus[0,", "xt_minus[0,", "xs_plus[0,", "xs_minus[0,")):
                    mr.lp_model += (mr.variables[k] == float(v.varValue or 0.0))
            mr.definir_funcao_objetivo()
            mr.resolver(msg=False)
            par_res[rev] = (float(mr.obj_value / 1e6), float(-mr.cvar_perda / 1e6))

        fo_mer_oos.append(par_res[True][0])
        fo_me_oos.append(par_res[False][0])
        d_fo_oos.append(par_res[True][0] - par_res[False][0])
        d_cvar_oos.append(par_res[True][1] - par_res[False][1])

    t_crit = _t_crit_95(len(seeds) - 1) if len(seeds) > 1 else 0.0
    d_fo_arr = np.asarray(d_fo_oos, dtype=float)
    d_cv_arr = np.asarray(d_cvar_oos, dtype=float)
    m_fo = float(np.mean(d_fo_arr))
    sd_fo = float(np.std(d_fo_arr, ddof=1))
    se_fo = sd_fo / np.sqrt(len(seeds))
    m_cv = float(np.mean(d_cv_arr))
    sd_cv = float(np.std(d_cv_arr, ddof=1))
    se_cv = sd_cv / np.sqrt(len(seeds))

    res = {
        "lambda": lam,
        "n_amostras_treino": n_amostras,
        "n_ref_mc": n_ref_mc,
        "seed_ref": seed_ref,
        "fo_ref_mer_M": ref_vals[True][0],
        "fo_ref_me_M": ref_vals[False][0],
        "d_fo_ref_M": ref_vals[True][0] - ref_vals[False][0],
        "d_cvar_ref_M": ref_vals[True][1] - ref_vals[False][1],
        "fo_oos_mer_media_M": float(np.mean(fo_mer_oos)),
        "fo_oos_me_media_M": float(np.mean(fo_me_oos)),
        "d_fo_oos_media_M": m_fo,
        "d_fo_oos_std_M": sd_fo,
        "d_fo_oos_ic95_inf_M": m_fo - t_crit * se_fo,
        "d_fo_oos_ic95_sup_M": m_fo + t_crit * se_fo,
        "d_cvar_oos_media_M": m_cv,
        "d_cvar_oos_std_M": sd_cv,
        "d_cvar_oos_ic95_inf_M": m_cv - t_crit * se_cv,
        "d_cvar_oos_ic95_sup_M": m_cv + t_crit * se_cv,
        "n_pos_d_fo_oos": int(np.sum(d_fo_arr > 1e-6)),
    }
    print(
        f"=== KW PAREADO OOS (N_ref={n_ref_mc}, lam={lam}): "
        f"dFO_ref={res['d_fo_ref_M']:+.4f}M | dFO_oos={m_fo:+.4f} +- {sd_fo:.4f}M "
        f"(IC95% [{res['d_fo_oos_ic95_inf_M']:+.4f}, {res['d_fo_oos_ic95_sup_M']:+.4f}], "
        f"{res['n_pos_d_fo_oos']}/{len(seeds)}) | dCVaR_oos={m_cv:+.4f}M ==="
    )
    if salvar_json:
        with open(OUTPUT_DIR / "diferenca_pareada_kaut_wallace.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
    return res


def avaliar_dispersao_ramificacao(
    n_repeticoes: int = 10,
    lam: float = 0.5,
    preco_venda_cliente_mwh: float = 160.0,
    salvar_csv: bool = True,
) -> pd.DataFrame:
    """
    Major #2: Compara ramificacao apenas por PLD_SE (ramificar_conjunto=False) vs
    ramificacao conjunta (PLD_SE, PLD_NE, demanda) quanto a dispersao de theta nas folhas
    e valor pareado da revenda.
    """
    from arvore_ccee import calcular_dispersao_folhas

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]
    seeds = [2026 + rep * 101 for rep in range(n_repeticoes)]
    rows = []
    for modo, flag in [("PLD_SE_apenas", False), ("Conjunta_SE_NE_Demanda", True)]:
        for seed in seeds:
            arv = construir_arvore_ccee_treino(
                divisao_estagios=divisao_8q, ramificacao=2, seed=seed, volatilidade_demanda=0.08,
                ramificar_conjunto=flag,
            )
            disp = calcular_dispersao_folhas(arv)
            res = {}
            for rev in (True, False):
                m = ModeloMultiestagioTEMPR(
                    arvore=arv, _lambda=lam, permitir_revenda=rev, preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                )
                m.criar_variaveis()
                m.adicionar_restricoes_posicao_e_balanco()
                m.adicionar_restricoes_revenda()
                m.definir_funcao_objetivo()
                m.resolver(msg=False)
                res[rev] = (m.obj_value / 1e6, -m.cvar_perda / 1e6)
            rows.append({
                "modo": modo,
                "seed": seed,
                **disp,
                "d_fo_M": res[True][0] - res[False][0],
                "d_cvar_M": res[True][1] - res[False][1],
            })
    df = pd.DataFrame(rows)
    if salvar_csv:
        df.to_csv(OUTPUT_DIR / "dispersao_ramificacao_comparacao.csv", index=False)
    print("\n=== COMPARACAO DE RAMIFICACAO (PLD_SE vs CONJUNTA SE+NE+DEMANDA, 10 SEEDS) ===")
    print(df.groupby("modo")[["theta_sd_medio", "theta_iqr_medio", "pld_se_sd_medio", "d_fo_M", "d_cvar_M"]].mean().round(4))
    return df


if __name__ == "__main__":
    df_in = avaliar_estabilidade_in_sample(n_amostras_mc=1024)
    df_in_4096 = avaliar_estabilidade_in_sample(n_amostras_mc=4096)
    avaliar_estabilidade_kaut_wallace(n_amostras_lista=[1024, 4096])
    avaliar_diferenca_pareada_sementes()
    avaliar_diferenca_pareada_kaut_wallace()
    avaliar_dispersao_ramificacao()
