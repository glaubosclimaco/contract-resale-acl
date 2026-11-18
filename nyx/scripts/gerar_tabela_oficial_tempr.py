"""
Script oficial de geração da tabela comparativa e experimentos consolidados para o artigo TEMPR.
- Árvore: 8 trimestres (Q0 a Q7), 24 meses reais da base CCEE (2021-2023) com ramificação conjunta.
- Preços a termo condicionais e spot calibrados estritamente na base de treino (2021-2023).
- Receita realista de clientes (P_venda = R$ 160/MWh), MWh reais (MWm * horas_mes).
- Incerteza de demanda por nó (volatilidade de 8%).
- Inclui benchmark 2S (Dois Estágios) e experimento regulatório do Decreto 5.163/2004 (lastro e piso).
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

OUTPUT_DIR = BASE / "resultados" / "tabela_tempr"
P_CLI_PADRAO = 160.0


def rodar_experimento_tabela(preco_venda_cliente_mwh: float = P_CLI_PADRAO):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = 2026
    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]

    print(f"Construindo árvore CCEE conjunta (8 trimestres, 24 meses) com seed={seed}...")
    arvore = construir_arvore_ccee_treino(
        divisao_estagios=divisao_8q,
        ramificacao=2,
        seed=seed,
        volatilidade_demanda=0.08,
    )
    print(f"Árvore gerada: {len(arvore.nodes)} nós, {len(arvore.leaves)} folhas, {len(arvore.stages)} estágios.")

    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    modos = ["com_revenda", "sem_revenda", "dois_estagios", "heuristica_95"]

    registros = []

    for lam in lambdas:
        for modo in modos:
            t0 = time.time()
            m = ModeloMultiestagioTEMPR(
                arvore=arvore,
                _lambda=lam,
                beta=0.95,
                permitir_revenda=(modo == "com_revenda"),
                preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                encargo_ccee_mwh=15.0,
                ano_inicio=2021,
                solver_name="HiGHS",
            )
            m.criar_variaveis()
            m.adicionar_restricoes_posicao_e_balanco()
            if modo == "heuristica_95":
                m.fixar_politica_heuristica_mercado(cobertura_pct=0.95)
            else:
                m.adicionar_restricoes_revenda()
                if modo == "dois_estagios":
                    m.fixar_politica_dois_estagios()
            m.definir_funcao_objetivo()
            status = m.resolver(msg=False)
            t_solve = time.time() - t0
            reg_met = m.calcular_metricas_regulatorias()
            mwh_total_m = float(np.sum((m.D_base["SE"] + m.D_base["NE"]) * m.horas_mes) / 1e6)

            registros.append({
                "lambda": lam,
                "modo": modo,
                "status": status,
                "fo_milhoes": m.obj_value / 1e6,
                "fo_r_mwh": (m.obj_value / 1e6) / mwh_total_m,
                "er_milhoes": m.expected_profit / 1e6,
                "cvar_milhoes": m.cvar_perda / 1e6,
                "cvar_profit_milhoes": -m.cvar_perda / 1e6,
                "var_milhoes": m.var_perda_alfa / 1e6,
                **reg_met,
                "tempo_s": t_solve,
                "n_vars": len(m.lp_model.variables()),
                "n_constr": len(m.lp_model.constraints),
            })
            print(
                f"lambda={lam:.2f} | {modo:14s} | FO={m.obj_value/1e6:7.2f}M | E[R]={m.expected_profit/1e6:7.2f}M "
                f"| CVaR5%={-m.cvar_perda/1e6:7.2f}M | Def12M={reg_met['deficit_12m_esp_gwh']:5.1f}GWh "
                f"| Gar95={reg_met['garantia_pico_2m_p95_M']:5.2f}M | t={t_solve*1000:5.1f}ms"
            )

    df = pd.DataFrame(registros)
    csv_path = OUTPUT_DIR / "tabela_experimentos_tempr.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}")

    # Tabela consolidada comparativa
    tabela_pivot = []
    for lam in lambdas:
        sub = df[df["lambda"] == lam].set_index("modo")
        er_rev = sub.loc["com_revenda", "er_milhoes"]
        cvar_rev = sub.loc["com_revenda", "cvar_milhoes"]
        fo_rev = sub.loc["com_revenda", "fo_milhoes"]

        er_srev = sub.loc["sem_revenda", "er_milhoes"]
        cvar_srev = sub.loc["sem_revenda", "cvar_milhoes"]
        fo_srev = sub.loc["sem_revenda", "fo_milhoes"]

        er_2s = sub.loc["dois_estagios", "er_milhoes"]
        cvar_2s = sub.loc["dois_estagios", "cvar_milhoes"]
        fo_2s = sub.loc["dois_estagios", "fo_milhoes"]

        er_heur = sub.loc["heuristica_95", "er_milhoes"]
        cvar_heur = sub.loc["heuristica_95", "cvar_milhoes"]
        fo_heur = sub.loc["heuristica_95", "fo_milhoes"]

        tabela_pivot.append({
            "lambda": lam,
            "ER_MER": er_rev,
            "CVaR_MER": cvar_rev,
            "FO_MER": fo_rev,
            "ER_ME": er_srev,
            "CVaR_ME": cvar_srev,
            "FO_ME": fo_srev,
            "ER_2S": er_2s,
            "CVaR_2S": cvar_2s,
            "FO_2S": fo_2s,
            "ER_HEUR": er_heur,
            "CVaR_HEUR": cvar_heur,
            "FO_HEUR": fo_heur,
            "dFO_revenda": fo_rev - fo_srev,
            "dER_revenda": er_rev - er_srev,
            "dCVaR_revenda": cvar_srev - cvar_rev,
            "dFO_vs_2S": fo_rev - fo_2s,
            "dFO_vs_heur": fo_rev - fo_heur,
            "dER_vs_heur": er_rev - er_heur,
            "dCVaR_vs_heur": cvar_heur - cvar_rev,
        })

    df_pivot = pd.DataFrame(tabela_pivot)
    pivot_path = OUTPUT_DIR / "comparativo_consolidado_tempr.csv"
    df_pivot.to_csv(pivot_path, index=False)
    print(f"Comparativo consolidado salvo em: {pivot_path}")
    return df_pivot


def rodar_experimento_regulatorio(
    n_repeticoes: int = 10,
    preco_venda_cliente_mwh: float = P_CLI_PADRAO,
    vr_mwh: float = 231.47,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Major #3 & #8: Experimento regulatório do Decreto 5.163/2004 (REN ANEEL 1.090/2024,
    VR CCEE 2025 atualizado = R$231,47/MWh, a partir de R$189,90/MWh base jul/2021 do Despacho 3.176/2024)
    e piso de contratação ex-ante de 100%, avaliado em 10 sementes (lam=0.5).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]
    seeds = [2026 + rep * 101 for rep in range(n_repeticoes)]

    regimes = [
        ("R0_Sem_Regra_Lastro", 0.0, 0.0),
        ("R1_Decreto5163_VR_231_47", vr_mwh, 0.0),
        ("R2_Piso_100pct_e_VR", vr_mwh, 1.0),
    ]
    estrategias = ["ME-R", "ME", "2S", "HEUR_95", "Unhedged"]
    rows = []

    print(f"\n=== EXPERIMENTO REGULATORIO: DECRETO 5.163/2004 & PISO 100% ({n_repeticoes} SEEDS, P_cli=R${preco_venda_cliente_mwh:.0f}) ===")
    for seed in seeds:
        arv = construir_arvore_ccee_treino(
            divisao_estagios=divisao_8q, ramificacao=2, seed=seed, volatilidade_demanda=0.08,
        )
        mwh_tot_m = float(np.sum((50.0 + 20.0) * ModeloMultiestagioTEMPR.HORAS_MES_PADRAO) / 1e6)

        for reg_nome, pen_vr, piso_cob in regimes:
            for est in estrategias:
                if piso_cob > 0.0 and est in ("HEUR_95", "Unhedged"):
                    # Sob piso rígido de 100%, HEUR_95 e Unhedged violam a regra ex-ante; avalia-se HEUR_100 para referência
                    if est == "HEUR_95":
                        cob_heur = 1.0
                    else:
                        continue
                else:
                    cob_heur = 0.95

                m = ModeloMultiestagioTEMPR(
                    arvore=arv,
                    _lambda=0.5,
                    beta=0.95,
                    permitir_revenda=(est == "ME-R"),
                    preco_venda_cliente_mwh=preco_venda_cliente_mwh,
                    encargo_ccee_mwh=15.0,
                    ano_inicio=2021,
                    penalidade_lastro_mwh=pen_vr,
                    cobertura_minima_lastro=(0.0 if est in ("HEUR_95", "Unhedged") else piso_cob),
                )
                m.criar_variaveis()
                m.adicionar_restricoes_posicao_e_balanco()
                if est == "HEUR_95":
                    m.fixar_politica_heuristica_mercado(cobertura_pct=cob_heur)
                    est_label = "HEUR_100" if piso_cob > 0.0 else "HEUR_95"
                elif est == "Unhedged":
                    for k, v in m.variables.items():
                        if k.startswith(("xm_", "xt_", "xs_")):
                            m.lp_model += (v == 0.0)
                    est_label = "Unhedged"
                else:
                    m.adicionar_restricoes_revenda()
                    if est == "2S":
                        m.fixar_politica_dois_estagios()
                    est_label = est

                m.definir_funcao_objetivo()
                status = m.resolver(msg=False)
                assert status == 1, f"Inviável em {reg_nome}/{est_label}/seed={seed}"
                met = m.calcular_metricas_regulatorias(vr_mwh=vr_mwh)

                rows.append({
                    "regime": reg_nome,
                    "seed": seed,
                    "estrategia": est_label,
                    "fo_M": m.obj_value / 1e6,
                    "fo_r_mwh": (m.obj_value / 1e6) / mwh_tot_m,
                    "er_M": m.expected_profit / 1e6,
                    "cvar_profit_M": -m.cvar_perda / 1e6,
                    **met,
                })

    df_det = pd.DataFrame(rows)
    df_res = df_det.groupby(["regime", "estrategia"], as_index=False)[
        ["fo_M", "fo_r_mwh", "er_M", "cvar_profit_M", "deficit_12m_esp_gwh", "penalidade_vr_implicita_M", "prob_incumprimento_12m_pct", "garantia_pico_2m_p95_M"]
    ].mean()

    # Diferenças pareadas ME-R vs ME dentro de cada regime
    piv = df_det.pivot_table(index=["regime", "seed"], columns="estrategia", values="fo_M").reset_index()
    piv["dFO_MER_ME_M"] = piv["ME-R"] - piv["ME"]
    piv_c = df_det.pivot_table(index=["regime", "seed"], columns="estrategia", values="cvar_profit_M").reset_index()
    piv["dCVaR_MER_ME_M"] = piv_c["ME-R"] - piv_c["ME"]
    piv_summary = piv.groupby("regime")[["dFO_MER_ME_M", "dCVaR_MER_ME_M"]].agg(["mean", "std"])

    df_det.to_csv(OUTPUT_DIR / "experimento_regulatorio_detalhado.csv", index=False)
    df_res.to_csv(OUTPUT_DIR / "experimento_regulatorio_lastro.csv", index=False)
    print(df_res.to_string(index=False))
    print("\n--- Ganho pareado ME-R - ME por Regime (10 seeds) ---")
    print(piv_summary.round(4))
    return df_det, df_res


if __name__ == "__main__":
    rodar_experimento_tabela()
    rodar_experimento_regulatorio()
