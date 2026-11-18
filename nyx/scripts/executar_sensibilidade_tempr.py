"""
Módulo de Análise de Sensibilidade Regulatória e Construção da Fronteira Eficiente (TEMPR).
Fase 5.

Sensibilidades Avaliadas:
1. Fronteira Eficiente: Retorno Esperado E[R] vs CVaR para lambda in [0.0, 0.1, ..., 1.0].
2. Sensibilidade aos Limites Regulatórios de Preço (PLD_max e PLD_min):
   - Cenário Base: Tetos históricos ANEEL (até R$ 1.470,57).
   - Cenário Teto Comprimido: Teto reduzido em 30% (limite estrutural ANEEL ~R$ 679/MWh).
   - Cenário Teto Elevado: Teto estendido.
3. Sensibilidade aos Encargos Setoriais (ESS/EER):
   - tau_reg in [0, 15, 30] R$/MWh.
4. Sensibilidade ao Bid-Ask Spread no Mercado a Termo:
   - spread_termo in [0.005, 0.015, 0.030] (0.5%, 1.5%, 3.0%).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

import sys
BASE = Path(__file__).resolve().parents[1]
for sub in ("config", "optimization", "optimization/models", "data_processing", "utils"):
    sys.path.append(str(BASE / sub))

from arvore_ccee import construir_arvore_ccee_treino
from modelo_multiestagio_tempr import ModeloMultiestagioTEMPR

OUTPUT_DIR = BASE / "resultados" / "sensibilidade_regulatoria"


def executar_sensibilidades():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]
    arv = construir_arvore_ccee_treino(
        divisao_estagios=divisao_8q,
        ramificacao=2,
        seed=2026,
        volatilidade_demanda=0.08,
    )

    # 1. Sensibilidade ao Encargo Setorial (ESS/EER)
    print("1. Avaliando sensibilidade aos Encargos Setoriais (0, 15, 30 R$/MWh)...")
    res_encargo = []
    for tau in [0.0, 15.0, 30.0]:
        for lam in [0.0, 0.5, 0.95]:
            m = ModeloMultiestagioTEMPR(arvore=arv, _lambda=lam, encargo_ccee_mwh=tau, permitir_revenda=True)
            m.criar_variaveis()
            m.adicionar_restricoes_posicao_e_balanco()
            m.adicionar_restricoes_revenda()
            m.definir_funcao_objetivo()
            m.resolver(msg=False)

            res_encargo.append({
                "encargo_mwh": tau,
                "lambda": lam,
                "fo_milhoes": m.obj_value / 1e6,
                "er_milhoes": m.expected_profit / 1e6,
                "cvar_milhoes": m.cvar_perda / 1e6,
            })
    pd.DataFrame(res_encargo).to_csv(OUTPUT_DIR / "sensibilidade_encargos.csv", index=False)

    # 2. Sensibilidade ao Bid-Ask Spread a Termo (incluindo base 1.0% para Fig. 4b)
    print("2. Avaliando sensibilidade ao Spread a Termo (0.5%, 1.0%, 1.5%, 3.0%)...")
    res_spread = []
    for sp in [0.005, 0.010, 0.015, 0.030]:
        for lam in [0.0, 0.5]:
            m = ModeloMultiestagioTEMPR(arvore=arv, _lambda=lam, spread_termo=sp, permitir_revenda=True)
            m.criar_variaveis()
            m.adicionar_restricoes_posicao_e_balanco()
            m.adicionar_restricoes_revenda()
            m.definir_funcao_objetivo()
            m.resolver(msg=False)

            res_spread.append({
                "spread_pct": sp * 100,
                "lambda": lam,
                "fo_milhoes": m.obj_value / 1e6,
                "er_milhoes": m.expected_profit / 1e6,
                "cvar_milhoes": m.cvar_perda / 1e6,
            })
    pd.DataFrame(res_spread).to_csv(OUTPUT_DIR / "sensibilidade_spread.csv", index=False)

    # 3. Sensibilidade ao Premio de Risco a Termo (M6: constante vs decaindo com o prazo)
    print("3. Avaliando sensibilidade ao Premio de Risco a Termo (-10% a +10%, constante e decaindo)...")
    for decaimento, nome_csv in [
        (False, "sensibilidade_premio_risco.csv"),
        (True, "sensibilidade_premio_risco_decaimento.csv"),
    ]:
        res_premio = []
        for prem in [-0.10, -0.05, 0.0, 0.05, 0.10]:
            for lam in [0.0, 0.5]:
                for modo in ["ME-R", "ME", "HEUR"]:
                    m = ModeloMultiestagioTEMPR(
                        arvore=arv,
                        _lambda=lam,
                        premio_risco_termo=prem,
                        decaimento_premio=decaimento,
                        permitir_revenda=(modo == "ME-R"),
                    )
                    m.criar_variaveis()
                    m.adicionar_restricoes_posicao_e_balanco()
                    if modo == "HEUR":
                        m.fixar_politica_heuristica_mercado(cobertura_pct=0.95)
                    else:
                        m.adicionar_restricoes_revenda()
                    m.definir_funcao_objetivo()
                    m.resolver(msg=False)

                    q_raiz_se = np.mean([m.variables[f"q[0,SE,{n}]"].varValue or 0.0 for n in range(3, 24)])
                    q_raiz_ne = np.mean([m.variables[f"q[0,NE,{n}]"].varValue or 0.0 for n in range(3, 24)])

                    buy_gwh = 0.0
                    sell_gwh = 0.0
                    for leaf in arv.leaves:
                        for nid in arv.caminho_ate_raiz(leaf.id):
                            nd = arv.nodes[nid]
                            n_min = max(nd.meses) + 1
                            for s in ["SE", "NE"]:
                                for n in range(n_min, 24):
                                    buy_gwh += leaf.prob * float(m.variables[f"xm_plus[{nid},{s},{n}]"].varValue or 0.0) * m.horas_mes[n] / 1e3
                                    sell_gwh += leaf.prob * float(m.variables[f"xm_minus[{nid},{s},{n}]"].varValue or 0.0) * m.horas_mes[n] / 1e3
                                for q in range(m.n_quarters):
                                    if q * 3 >= n_min:
                                        hq = float(np.sum(m.horas_mes[q * 3:q * 3 + 3]))
                                        buy_gwh += leaf.prob * float(m.variables[f"xt_plus[{nid},{s},{q}]"].varValue or 0.0) * hq / 1e3
                                        sell_gwh += leaf.prob * float(m.variables[f"xt_minus[{nid},{s},{q}]"].varValue or 0.0) * hq / 1e3
                                for h in range(m.n_semesters):
                                    if h * 6 >= n_min:
                                        hh = float(np.sum(m.horas_mes[h * 6:h * 6 + 6]))
                                        buy_gwh += leaf.prob * float(m.variables[f"xs_plus[{nid},{s},{h}]"].varValue or 0.0) * hh / 1e3
                                        sell_gwh += leaf.prob * float(m.variables[f"xs_minus[{nid},{s},{h}]"].varValue or 0.0) * hh / 1e3

                    res_premio.append({
                        "premio_pct": prem * 100,
                        "decaimento": decaimento,
                        "lambda": lam,
                        "modo": modo,
                        "fo_milhoes": m.obj_value / 1e6,
                        "er_milhoes": m.expected_profit / 1e6,
                        "cvar_profit_milhoes": -m.cvar_perda / 1e6,
                        "q_raiz_se_mwm": float(q_raiz_se),
                        "q_raiz_ne_mwm": float(q_raiz_ne),
                        "compras_esp_gwh": float(buy_gwh),
                        "revendas_esp_gwh": float(sell_gwh),
                    })
        pd.DataFrame(res_premio).to_csv(OUTPUT_DIR / nome_csv, index=False)

    # 4. Sensibilidade a Tarifa de Venda P_cli (Major #4: calibracao economica realista vs margem alta)
    print("4. Avaliando sensibilidade a Tarifa de Venda P_cli (150, 160, 180, 260 R$/MWh)...")
    res_tarifa = []
    mwh_tot_m = float(np.sum((50.0 + 20.0) * arv.nodes[0].pld_se * 0.0 + 70.0 * np.array(
        [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744] * 2, dtype=float
    )) / 1e6)
    for p_cli in [150.0, 160.0, 180.0, 260.0]:
        for lam in [0.0, 0.5]:
            vals = {}
            for modo, rev in [("ME-R", True), ("ME", False)]:
                m = ModeloMultiestagioTEMPR(
                    arvore=arv, _lambda=lam, preco_venda_cliente_mwh=p_cli, permitir_revenda=rev,
                )
                m.criar_variaveis()
                m.adicionar_restricoes_posicao_e_balanco()
                m.adicionar_restricoes_revenda()
                m.definir_funcao_objetivo()
                m.resolver(msg=False)
                vals[modo] = (m.obj_value / 1e6, m.expected_profit / 1e6, -m.cvar_perda / 1e6)
            d_fo = vals["ME-R"][0] - vals["ME"][0]
            d_cvar = vals["ME-R"][2] - vals["ME"][2]
            res_tarifa.append({
                "p_cli_mwh": p_cli,
                "lambda": lam,
                "fo_mer_M": vals["ME-R"][0],
                "fo_me_M": vals["ME"][0],
                "er_mer_M": vals["ME-R"][1],
                "cvar_mer_M": vals["ME-R"][2],
                "cvar_me_M": vals["ME"][2],
                "dFO_M": d_fo,
                "dFO_r_mwh": d_fo / mwh_tot_m,
                "dFO_pct_fo": (d_fo / abs(vals["ME"][0]) * 100.0) if abs(vals["ME"][0]) > 1e-6 else 0.0,
                "dCVaR_M": d_cvar,
            })
    pd.DataFrame(res_tarifa).to_csv(OUTPUT_DIR / "sensibilidade_tarifa.csv", index=False)

    # 5. Resumo empirico de profundidade BBCE 2023 por maturidade (Minor #13)
    liq_rows = []
    for prod, fname in [("Mensal", "liquidez_mensal.csv"), ("Trimestral", "liquidez_trimestral.csv"), ("Semestral", "liquidez_semestral.csv")]:
        fpath = BASE / "data" / fname
        if fpath.exists():
            df_l = pd.read_csv(fpath, index_col=0)
            for lag, row in df_l.iterrows():
                vals_v = row.dropna().to_numpy(dtype=float)
                if len(vals_v) > 0:
                    liq_rows.append({
                        "produto": prod,
                        "lag_meses": int(lag),
                        "n_meses_obs": int(len(vals_v)),
                        "media_mwmed": float(np.mean(vals_v)),
                        "mediana_mwmed": float(np.median(vals_v)),
                        "max_mwmed": float(np.max(vals_v)),
                    })
    if liq_rows:
        pd.DataFrame(liq_rows).to_csv(OUTPUT_DIR / "resumo_liquidez_bbce_2023.csv", index=False)

    print("Sensibilidades concluídas e salvas em resultados/sensibilidade_regulatoria/.")


if __name__ == "__main__":
    executar_sensibilidades()
