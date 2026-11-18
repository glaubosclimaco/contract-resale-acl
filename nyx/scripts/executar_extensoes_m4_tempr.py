"""
Extensões Estruturais (M4) para o Artigo TEMPR:
1. Modulação Horária Intra-Mensal: perfis de carga (plano, industrial, comercial, residencial)
   contra contratos flat sobre as 26.280h de treino (2021-2023) e 17.544h de teste (2024-2025).
2. Contratação Cruzada NE->SE e Risco de Base (FTR): avalia o uso de contratos no NE para
   cobrir carga no SE sob exposição ao diferencial PLD_SE - PLD_NE e proteção via FTR.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
for sub in ("config", "optimization", "optimization/models", "data_processing", "utils"):
    sys.path.append(str(BASE / sub))

from arvore_ccee import construir_arvore_ccee_treino
from modelo_multiestagio_tempr import ModeloMultiestagioTEMPR
from perfis_horarios import PERFIS_CONSUMO_SINTETICOS

DATA_DIR = BASE / "data" / "processados_tempr"
OUTPUT_DIR = BASE / "resultados" / "sensibilidade_regulatoria"


def calcular_fatores_captura_mensais(csv_path: Path, n_meses: int = 24) -> dict[str, dict[str, np.ndarray]]:
    """
    Calcula os fatores de captura gamma_{s,m} = mean_h(w_h * PLD_{s,h}) / mean_h(PLD_{s,h})
    para os primeiros `n_meses` (ou ciclo sazonal 12 meses repetido) de um CSV horário CCEE.
    """
    df = pd.read_csv(csv_path)
    df["MES_CAL"] = df["MES_REFERENCIA"] % 100
    resultados = {}

    for nome, vals in PERFIS_CONSUMO_SINTETICOS.items():
        w = np.asarray(vals, dtype=float)
        w = w / np.mean(w)  # média diária = 1.0 (mesma energia média mensal)
        df["_w"] = df["HORA"].astype(int).map(lambda h: w[h])

        # Para calibração na árvore (2021-2023), usar fator médio por mês calendário (1..12) repetido em 24m
        # Para teste 2024-2025, usar exatamente os 24 meses sequenciais
        meses_unicos = sorted(df["MES_REFERENCIA"].unique())
        if len(meses_unicos) == n_meses:
            g_se, g_ne = [], []
            for mr in meses_unicos:
                sub = df[df["MES_REFERENCIA"] == mr]
                g_se.append(float((sub["PLD_SE"] * sub["_w"]).mean() / sub["PLD_SE"].mean()))
                g_ne.append(float((sub["PLD_NE"] * sub["_w"]).mean() / sub["PLD_NE"].mean()))
            resultados[nome] = {"SE": np.array(g_se), "NE": np.array(g_ne)}
        else:
            g_se_cal = {}
            g_ne_cal = {}
            for mc in range(1, 13):
                sub = df[df["MES_CAL"] == mc]
                g_se_cal[mc] = float((sub["PLD_SE"] * sub["_w"]).mean() / sub["PLD_SE"].mean())
                g_ne_cal[mc] = float((sub["PLD_NE"] * sub["_w"]).mean() / sub["PLD_NE"].mean())
            resultados[nome] = {
                "SE": np.array([g_se_cal[(m % 12) + 1] for m in range(n_meses)], dtype=float),
                "NE": np.array([g_ne_cal[(m % 12) + 1] for m in range(n_meses)], dtype=float),
            }
    return resultados


def executar_modulacao_horaria(arv) -> pd.DataFrame:
    """Avalia in-sample (2021-2023) e impacto horário real (2024-2025) por perfil de carga."""
    gammas_tr = calcular_fatores_captura_mensais(DATA_DIR / "pld_horario_treino_2021_2023.csv", n_meses=24)
    gammas_te = calcular_fatores_captura_mensais(DATA_DIR / "pld_horario_teste_2024_2025.csv", n_meses=24)

    df_te = pd.read_csv(DATA_DIR / "pld_horario_teste_2024_2025.csv")
    mensal_te = df_te.groupby("MES_REFERENCIA").agg(
        PLD_SE=("PLD_SE", "mean"),
        PLD_NE=("PLD_NE", "mean"),
        horas=("HORA", "count"),
    ).sort_index()
    pld_te_se = mensal_te["PLD_SE"].to_numpy()
    pld_te_ne = mensal_te["PLD_NE"].to_numpy()
    horas_te = mensal_te["horas"].to_numpy()

    m_plano = ModeloMultiestagioTEMPR(
        arvore=arv,
        _lambda=0.5,
        permitir_revenda=True,
        fator_captura_demanda=gammas_tr["plano"],
    )
    m_plano.criar_variaveis()
    m_plano.adicionar_restricoes_posicao_e_balanco()
    m_plano.adicionar_restricoes_revenda()
    m_plano.definir_funcao_objetivo()
    m_plano.resolver(msg=False)
    x_plano = {
        k: float(v.varValue or 0.0)
        for k, v in m_plano.variables.items()
        if k.startswith(("xm_", "xt_", "xs_"))
    }

    registros = []
    for perfil in ["plano", "industrial", "comercial", "residencial"]:
        m = ModeloMultiestagioTEMPR(
            arvore=arv,
            _lambda=0.5,
            permitir_revenda=True,
            fator_captura_demanda=gammas_tr[perfil],
        )
        m.criar_variaveis()
        m.adicionar_restricoes_posicao_e_balanco()
        m.adicionar_restricoes_revenda()
        m.definir_funcao_objetivo()
        m.resolver(msg=False)

        # Avalia exatamente a carteira otima do caso plano sob o gamma do perfil
        m_fix = ModeloMultiestagioTEMPR(
            arvore=arv,
            _lambda=0.5,
            permitir_revenda=True,
            fator_captura_demanda=gammas_tr[perfil],
        )
        m_fix.criar_variaveis()
        m_fix.adicionar_restricoes_posicao_e_balanco()
        m_fix.adicionar_restricoes_revenda()
        for k, val in x_plano.items():
            m_fix.lp_model += (m_fix.variables[k] == val)
        m_fix.definir_funcao_objetivo()
        m_fix.resolver(msg=False)

        # Custo/economia ex-post da modulação horária no teste real 2024-2025 frente ao caso plano (50 MWm SE, 20 MWm NE)
        delta_oos_se = -np.sum((gammas_te[perfil]["SE"] - 1.0) * pld_te_se * 50.0 * horas_te) / 1e6
        delta_oos_ne = -np.sum((gammas_te[perfil]["NE"] - 1.0) * pld_te_ne * 20.0 * horas_te) / 1e6

        registros.append({
            "perfil": perfil,
            "gamma_tr_SE_medio": float(np.mean(gammas_tr[perfil]["SE"])),
            "gamma_tr_NE_medio": float(np.mean(gammas_tr[perfil]["NE"])),
            "gamma_te_SE_medio": float(np.mean(gammas_te[perfil]["SE"])),
            "gamma_te_NE_medio": float(np.mean(gammas_te[perfil]["NE"])),
            "fo_in_sample_M": m.obj_value / 1e6,
            "fo_carteira_plana_M": m_fix.obj_value / 1e6,
            "ganho_reotimizacao_M": (m.obj_value - m_fix.obj_value) / 1e6,
            "er_in_sample_M": m.expected_profit / 1e6,
            "cvar_profit_in_sample_M": -m.cvar_perda / 1e6,
            "delta_oos_24_25_M": float(delta_oos_se + delta_oos_ne),
        })
    df_out = pd.DataFrame(registros)
    df_out.to_csv(OUTPUT_DIR / "extensao_m4_modulacao_horaria.csv", index=False)
    print("\n=== M4-A: MODULAÇÃO HORÁRIA INTRA-MENSAL (CONTRATOS FLAT vs PERFIL DE CARGA) ===")
    print(df_out.to_string(index=False))
    return df_out


def executar_contrato_cruzado_ne_se(arv) -> pd.DataFrame:
    """Avalia contratação cruzada NE->SE com risco de base e opção de FTR na árvore."""
    # Spread positivo esperado nos nós da árvore (base de precificação de FTR justo)
    spread_pos_esp = []
    for leaf in arv.leaves:
        sp = np.maximum(0.0, leaf.pld_se - leaf.pld_ne)
        spread_pos_esp.append(leaf.prob * float(np.mean(sp)))
    ftr_justo = float(np.sum(spread_pos_esp))

    configs = [
        ("Sem_Cruzado_Base", 0.5, False, None),
        ("Cruzado_NE_SE_Sem_FTR_lam0", 0.0, True, None),
        ("Cruzado_NE_SE_Sem_FTR_lam05", 0.5, True, None),
        ("Cruzado_NE_SE_FTR_Justo_0", 0.5, True, 0.0),
        ("Cruzado_NE_SE_FTR_Premio_1", 0.5, True, 1.0),
        ("Cruzado_NE_SE_FTR_Premio_5", 0.5, True, 5.0),
    ]

    registros = []
    for nome, lam, cross, ftr_c in configs:
        m = ModeloMultiestagioTEMPR(
            arvore=arv,
            _lambda=lam,
            permitir_revenda=True,
            permitir_contrato_cruzado=cross,
            custo_ftr_mwh=ftr_c,
        )
        m.criar_variaveis()
        m.adicionar_restricoes_posicao_e_balanco()
        m.adicionar_restricoes_revenda()
        m.definir_funcao_objetivo()
        m.resolver(msg=False)

        qc_esp_gwh = 0.0
        qftr_esp_gwh = 0.0
        qse_esp_gwh = 0.0
        for leaf in arv.leaves:
            caminho = arv.caminho_ate_raiz(leaf.id)
            for n in range(3, arv.n_months):
                anc = caminho[n // 3]
                h_n = float(m.horas_mes[n])
                v_se = m.variables[f"q[{anc},SE,{n}]"].varValue or 0.0
                qse_esp_gwh += leaf.prob * v_se * h_n / 1e3
                if cross:
                    v_c = m.variables[f"qc[{anc},{n}]"].varValue or 0.0
                    qc_esp_gwh += leaf.prob * v_c * h_n / 1e3
                if ftr_c is not None:
                    v_f = m.variables[f"qftr[{anc},{n}]"].varValue or 0.0
                    qftr_esp_gwh += leaf.prob * v_f * h_n / 1e3

        registros.append({
            "cenario": nome,
            "lambda": lam,
            "custo_ftr_mwh": ftr_c if ftr_c is not None else np.nan,
            "fo_M": m.obj_value / 1e6,
            "er_M": m.expected_profit / 1e6,
            "cvar_profit_M": -m.cvar_perda / 1e6,
            "q_SE_esp_GWh": float(qse_esp_gwh),
            "q_cross_NE_SE_esp_GWh": float(qc_esp_gwh),
            "q_ftr_esp_GWh": float(qftr_esp_gwh),
        })

    df_out = pd.DataFrame(registros)
    df_out.to_csv(OUTPUT_DIR / "extensao_m4_contrato_cruzado_ftr.csv", index=False)
    print(f"\n=== M4-B: CONTRATAÇÃO CRUZADA NE->SE E FTR (Spread pos. esperado arvore = R$ {ftr_justo:.2f}/MWh) ===")
    print(df_out.to_string(index=False))
    return df_out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]
    arv = construir_arvore_ccee_treino(divisao_estagios=divisao_8q, ramificacao=2, seed=2026)
    executar_modulacao_horaria(arv)
    executar_contrato_cruzado_ne_se(arv)


if __name__ == "__main__":
    main()
