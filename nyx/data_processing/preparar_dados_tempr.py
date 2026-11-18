"""
Módulo de processamento de dados CCEE para o artigo TEMPR.
- Submercados: SUDESTE/CENTRO-OESTE (SE) e NORDESTE (NE).
- Separação Temporal:
    * Treino/Calibração: 2021-2023 (8.760 h * 3 = 26.280 h)
    * Teste Fora da Amostra: 2024-2025 (8.784 h + 8.760 h = 17.544 h)
- Parâmetros Regulatórios:
    * Limites de PLD (teto e piso) por ano.
    * Encargos regulatórios médios (ESS/EER) = R$ 15,00/MWh.
    * Carteira de Estudo de Caso: 50 MWmédios no SE e 20 MWmédios no NE.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "nyx" / "data"
OUTPUT_DIR = DATA_DIR / "processados_tempr"


def carregar_dados_brutos() -> pd.DataFrame:
    dfs = []
    for ano in range(2021, 2026):
        arq = DATA_DIR / f"pld_horario_{ano}.csv"
        if not arq.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {arq}")
        df_ano = pd.read_csv(arq, sep=";")
        df_ano["ANO"] = ano
        dfs.append(df_ano)
    
    df_total = pd.concat(dfs, ignore_index=True)
    # Filtrar submercados de interesse
    df_sub = df_total[df_total["SUBMERCADO"].isin(["SUDESTE", "NORDESTE"])].copy()
    
    # Criar timestamp/datetime aproximado para indexação temporal
    df_sub["DATA"] = pd.to_datetime(
        df_sub["ANO"].astype(str) + "-" + 
        df_sub["MES_REFERENCIA"].astype(str).str[-2:] + "-" + 
        df_sub["DIA"].astype(str).str.zfill(2) + " " + 
        df_sub["HORA"].astype(str).str.zfill(2) + ":00:00"
    )
    df_sub = df_sub.sort_values(["DATA", "SUBMERCADO"]).reset_index(drop=True)
    return df_sub


def processar_e_salvar():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Carregando séries horárias brutas da CCEE...")
    df = carregar_dados_brutos()

    # Pivotar para colunas SE e NE
    df_pivot = df.pivot_table(
        index=["DATA", "ANO", "MES_REFERENCIA", "DIA", "HORA"],
        columns="SUBMERCADO",
        values="PLD_HORA"
    ).reset_index()
    df_pivot.rename(columns={"SUDESTE": "PLD_SE", "NORDESTE": "PLD_NE"}, inplace=True)
    df_pivot["SPREAD_SE_NE"] = df_pivot["PLD_SE"] - df_pivot["PLD_NE"]

    # Divisão Treino e Teste
    treino = df_pivot[df_pivot["ANO"].isin([2021, 2022, 2023])].copy()
    teste = df_pivot[df_pivot["ANO"].isin([2024, 2025])].copy()

    treino_path = OUTPUT_DIR / "pld_horario_treino_2021_2023.csv"
    teste_path = OUTPUT_DIR / "pld_horario_teste_2024_2025.csv"

    treino.to_csv(treino_path, index=False)
    teste.to_csv(teste_path, index=False)
    print(f"Treino salvo ({len(treino)} registros) em: {treino_path}")
    print(f"Teste salvo ({len(teste)} registros) em: {teste_path}")

    # Parâmetros Regulatórios Consolidados
    parametros = {
        "submercados": ["SUDESTE", "NORDESTE"],
        "anos_treino": [2021, 2022, 2023],
        "anos_teste": [2024, 2025],
        "encargo_regulatorio_mwh": 15.00,  # ESS + EER médio estimado
        "limites_pld": {
            "2021": {"min": 49.77, "max_estrutural": 583.88, "max_horario": 1197.87},
            "2022": {"min": 55.70, "max_estrutural": 646.58, "max_horario": 1326.50},
            "2023": {"min": 69.04, "max_estrutural": 684.73, "max_horario": 1404.77},
            "2024": {"min": 61.07, "max_estrutural": 716.80, "max_horario": 1470.57},
            "2025": {"min": 58.60, "max_estrutural": 751.73, "max_horario": 1542.23}
        },
        "carteira_estudo_caso": {
            "sudeste_mwmed": 50.0,
            "nordeste_mwmed": 20.0,
            "perfil_consumo": "comercial_industrial_diurno",
            "fator_ponta": 1.25,
            "fator_fora_ponta": 0.90
        },
        "heuristicas_mercado": {
            "hedge_estatico_pct": 0.95,  # 95% do consumo contratado ex-ante no longo prazo
            "tolerancia_spot_pct": 0.05
        }
    }

    params_path = OUTPUT_DIR / "parametros_regulatorios_tempr.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(parametros, f, indent=4, ensure_ascii=False)
    print(f"Parâmetros salvos em: {params_path}")

    # Estatísticas Descritivas para a Tabela do Artigo
    resumo_treino = treino[["PLD_SE", "PLD_NE", "SPREAD_SE_NE"]].describe().T
    resumo_teste = teste[["PLD_SE", "PLD_NE", "SPREAD_SE_NE"]].describe().T
    
    resumo_treino["Amostra"] = "Treino (2021-2023)"
    resumo_teste["Amostra"] = "Teste (2024-2025)"
    
    resumo_geral = pd.concat([resumo_treino, resumo_teste])
    resumo_path = OUTPUT_DIR / "resumo_estatistico_pld.csv"
    resumo_geral.to_csv(resumo_path)
    print(f"Resumo estatístico salvo em: {resumo_path}")

    return treino, teste, parametros


if __name__ == "__main__":
    processar_e_salvar()
