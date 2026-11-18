"""
Módulo de Cálculo Formal de VSS e EVPI In-Sample (para o Artigo TEMPR).

NOTA DE LEGADO (OOS):
A função `simular_politica_no_historico_real` e o CSV `backtesting_historico_real_2024_2025.csv`
representam apenas um replay estático legado da decisão do 1º estágio e NÃO são usados no artigo.
O protocolo oficial fora da amostra (2024-2025) reportado no artigo está em
`nyx/scripts/backtest_rolante_tempr.py` (horizonte encolhendo em janela expansiva).

Métricas Clássicas de Programação Estocástica geradas por este script (`metricas_estocasticas_vss_evpi.json`):
- VSS (Value of Stochastic Solution): VSS = RP - EEV (em maximização).
- EVPI (Expected Value of Perfect Information): EVPI = WS - RP.
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
from arvore_cenarios import ArvoreCenarios, NoCenario
from modelo_multiestagio_tempr import ModeloMultiestagioTEMPR

OUTPUT_DIR = BASE / "resultados" / "backtesting_out_of_sample"


def carregar_dados_teste_2024_2025():
    csv_teste = BASE / "data" / "processados_tempr" / "pld_horario_teste_2024_2025.csv"
    if not csv_teste.exists():
        csv_teste = BASE / "nyx" / "data" / "processados_tempr" / "pld_horario_teste_2024_2025.csv"
    df = pd.read_csv(csv_teste)
    
    # Agregar mensalmente médias de PLD e horas por mês
    grp = df.groupby(["ANO", "MES_REFERENCIA"])
    resumo = grp.agg({
        "PLD_SE": "mean",
        "PLD_NE": "mean",
        "HORA": "count"
    }).reset_index().rename(columns={"HORA": "horas"})
    
    # 24 meses ordenados
    resumo = resumo.sort_values("MES_REFERENCIA").reset_index(drop=True)
    return resumo


def resolver_problema_deterministico_cenario_medio(arvore_treino, _lambda=0.5):
    """
    Constrói e resolve o problema determinístico do cenário médio (EEV setup):
    Árvore com as mesmas 8 etapas trimestrais contendo o cenário médio determinístico.
    """
    n_months = arvore_treino.n_months
    divisao_8q = arvore_treino.divisao_estagios
    # Médias ponderadas dos nós em cada mês
    pld_medio_se = np.zeros(n_months)
    pld_medio_ne = np.zeros(n_months)
    dem_media = np.zeros(n_months)
    for leaf in arvore_treino.leaves:
        pld_medio_se += leaf.prob * leaf.pld_se
        pld_medio_ne += leaf.prob * leaf.pld_ne
        dem_media += leaf.prob * leaf.demanda_fator

    nodes_det = {}
    stages_det = [[i] for i in range(len(divisao_8q))]
    for q in range(len(divisao_8q)):
        parent = (q - 1) if q > 0 else None
        meses_q = divisao_8q[q]
        nodes_det[q] = NoCenario(
            id=q, parent_id=parent, stage=q, meses=meses_q, prob=1.0, cond_prob=1.0,
            cenarios_indices=[0], pld=pld_medio_se.copy(), pld_se=pld_medio_se.copy(),
            pld_ne=pld_medio_ne.copy(), demanda_fator=dem_media.copy()
        )
        if parent is not None:
            nodes_det[parent].children_ids.append(q)

    arvore_det = ArvoreCenarios(
        nodes=nodes_det, stages=stages_det, n_months=n_months, n_series_originais=1,
        divisao_estagios=divisao_8q
    )
    m_det = ModeloMultiestagioTEMPR(arvore_det, _lambda=_lambda, permitir_revenda=False)
    m_det.criar_variaveis()
    m_det.adicionar_restricoes_posicao_e_balanco()
    m_det.adicionar_restricoes_revenda()
    m_det.definir_funcao_objetivo()
    m_det.resolver(msg=False)
    return m_det


def resolver_wait_and_see(arvore_treino, _lambda=0.5):
    """
    Calcula a solução Wait-and-See (WS):
    Resolve cada cenário folha isoladamente ao longo de seu caminho temporal com a mesma regra de preço.
    WS = sum_{ell} prob_{ell} * Lucro(ell)
    """
    n_months = arvore_treino.n_months
    divisao_8q = arvore_treino.divisao_estagios
    ws_lucro_esperado = 0.0

    for leaf in arvore_treino.leaves:
        caminho = arvore_treino.caminho_ate_raiz(leaf.id)
        nodes_ws = {}
        stages_ws = [[i] for i in range(len(caminho))]
        for idx, nid in enumerate(caminho):
            orig_node = arvore_treino.nodes[nid]
            parent = (idx - 1) if idx > 0 else None
            nodes_ws[idx] = NoCenario(
                id=idx, parent_id=parent, stage=idx, meses=orig_node.meses, prob=1.0, cond_prob=1.0,
                cenarios_indices=[leaf.id], pld=orig_node.pld_se.copy(), pld_se=orig_node.pld_se.copy(),
                pld_ne=orig_node.pld_ne.copy(), demanda_fator=orig_node.demanda_fator.copy()
            )
            if parent is not None:
                nodes_ws[parent].children_ids.append(idx)

        arv_cen = ArvoreCenarios(
            nodes=nodes_ws, stages=stages_ws, n_months=n_months, n_series_originais=1,
            divisao_estagios=divisao_8q
        )
        m_cen = ModeloMultiestagioTEMPR(arv_cen, _lambda=_lambda, permitir_revenda=True)
        m_cen.criar_variaveis()
        m_cen.adicionar_restricoes_posicao_e_balanco()
        m_cen.adicionar_restricoes_revenda()
        m_cen.definir_funcao_objetivo()
        m_cen.resolver(msg=False)
        ws_lucro_esperado += leaf.prob * m_cen.expected_profit

    return ws_lucro_esperado


def simular_politica_no_historico_real(m_modelo, dados_teste_df, nome_politica=""):
    """
    Simula a execução da política do 1º estágio de um modelo sobre o histórico real de 2024-2025.
    """
    raiz_id = 0
    horas = dados_teste_df["horas"].to_numpy(dtype=float)
    pld_real_se = dados_teste_df["PLD_SE"].to_numpy(dtype=float)
    pld_real_ne = dados_teste_df["PLD_NE"].to_numpy(dtype=float)
    
    def _get_v(k: str) -> float:
        var = m_modelo.variables.get(k)
        return float(var.varValue) if (var is not None and var.varValue is not None) else 0.0

    # Contratos decididos no 1º estágio (nó raiz)
    compras_m_se = np.array([_get_v(f"xm_plus[{raiz_id},SE,{n}]") for n in range(24)])
    compras_m_ne = np.array([_get_v(f"xm_plus[{raiz_id},NE,{n}]") for n in range(24)])
    vendas_m_se = np.array([_get_v(f"xm_minus[{raiz_id},SE,{n}]") for n in range(24)])
    vendas_m_ne = np.array([_get_v(f"xm_minus[{raiz_id},NE,{n}]") for n in range(24)])

    compras_t_se = np.array([_get_v(f"xt_plus[{raiz_id},SE,{q}]") for q in range(8)])
    compras_t_ne = np.array([_get_v(f"xt_plus[{raiz_id},NE,{q}]") for q in range(8)])
    vendas_t_se = np.array([_get_v(f"xt_minus[{raiz_id},SE,{q}]") for q in range(8)])
    vendas_t_ne = np.array([_get_v(f"xt_minus[{raiz_id},NE,{q}]") for q in range(8)])

    compras_s_se = np.array([_get_v(f"xs_plus[{raiz_id},SE,{h}]") for h in range(4)])
    compras_s_ne = np.array([_get_v(f"xs_plus[{raiz_id},NE,{h}]") for h in range(4)])
    vendas_s_se = np.array([_get_v(f"xs_minus[{raiz_id},SE,{h}]") for h in range(4)])
    vendas_s_ne = np.array([_get_v(f"xs_minus[{raiz_id},NE,{h}]") for h in range(4)])

    # Expandir trimestrais e semestrais para meses
    compras_t_mes_se = np.repeat(compras_t_se, 3)
    compras_t_mes_ne = np.repeat(compras_t_ne, 3)
    vendas_t_mes_se = np.repeat(vendas_t_se, 3)
    vendas_t_mes_ne = np.repeat(vendas_t_ne, 3)

    compras_s_mes_se = np.repeat(compras_s_se, 6)
    compras_s_mes_ne = np.repeat(compras_s_ne, 6)
    vendas_s_mes_se = np.repeat(vendas_s_se, 6)
    vendas_s_mes_ne = np.repeat(vendas_s_ne, 6)

    posicao_contratada_se = (
        compras_m_se + compras_t_mes_se + compras_s_mes_se
        - vendas_m_se - vendas_t_mes_se - vendas_s_mes_se
    )
    posicao_contratada_ne = (
        compras_m_ne + compras_t_mes_ne + compras_s_mes_ne
        - vendas_m_ne - vendas_t_mes_ne - vendas_s_mes_ne
    )

    demanda_se = m_modelo.D_base["SE"]
    demanda_ne = m_modelo.D_base["NE"]

    # 1. Receita de suprimento a clientes (R$ 260/MWh)
    rec_clientes = np.sum(m_modelo.preco_venda_cliente_mwh * (demanda_se + demanda_ne) * horas)
    
    # 2. Encargos setoriais ESS/EER (R$ 15/MWh)
    encargos = np.sum(m_modelo.encargo_ccee_mwh * (demanda_se + demanda_ne) * horas)

    # 3. Custo líquido de contratos a termo firmados no 1º estágio
    pm_c_se = m_modelo.precos_termo[raiz_id]["SE"]["pm_c"]
    pm_c_ne = m_modelo.precos_termo[raiz_id]["NE"]["pm_c"]
    pm_v_se = m_modelo.precos_termo[raiz_id]["SE"]["pm_v"]
    pm_v_ne = m_modelo.precos_termo[raiz_id]["NE"]["pm_v"]

    custo_contratos = (
        np.sum(pm_c_se * compras_m_se * horas) + np.sum(pm_c_ne * compras_m_ne * horas)
        - np.sum(pm_v_se * vendas_m_se * horas) - np.sum(pm_v_ne * vendas_m_ne * horas)
    )
    for q in range(8):
        h_q = np.sum(horas[q * 3:q * 3 + 3])
        custo_contratos += m_modelo.precos_termo[raiz_id]["SE"]["pt_c"][q] * compras_t_se[q] * h_q
        custo_contratos += m_modelo.precos_termo[raiz_id]["NE"]["pt_c"][q] * compras_t_ne[q] * h_q
        custo_contratos -= m_modelo.precos_termo[raiz_id]["SE"]["pt_v"][q] * vendas_t_se[q] * h_q
        custo_contratos -= m_modelo.precos_termo[raiz_id]["NE"]["pt_v"][q] * vendas_t_ne[q] * h_q
    for h in range(4):
        h_h = np.sum(horas[h * 6:h * 6 + 6])
        custo_contratos += m_modelo.precos_termo[raiz_id]["SE"]["ps_c"][h] * compras_s_se[h] * h_h
        custo_contratos += m_modelo.precos_termo[raiz_id]["NE"]["ps_c"][h] * compras_s_ne[h] * h_h
        custo_contratos -= m_modelo.precos_termo[raiz_id]["SE"]["ps_v"][h] * vendas_s_se[h] * h_h
        custo_contratos -= m_modelo.precos_termo[raiz_id]["NE"]["ps_v"][h] * vendas_s_ne[h] * h_h

    # 4. Liquidação spot física nos preços reais de 2024-2025
    sobra_se = posicao_contratada_se - demanda_se
    sobra_ne = posicao_contratada_ne - demanda_ne

    liq_spot = np.sum(pld_real_se * sobra_se * horas) + np.sum(pld_real_ne * sobra_ne * horas)

    lucro_total = rec_clientes - encargos - custo_contratos + liq_spot
    return {
        "politica": nome_politica,
        "lucro_total_milhoes": lucro_total / 1e6,
        "receita_clientes_milhoes": rec_clientes / 1e6,
        "custo_contratos_milhoes": custo_contratos / 1e6,
        "liquidacao_spot_milhoes": liq_spot / 1e6,
        "encargos_milhoes": encargos / 1e6,
    }


def executar_backtesting_e_metricas():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Iniciando Fase 4: Backtesting Real Fora da Amostra (2024-2025)...")

    # 1. Carregar dados de teste
    dados_teste = carregar_dados_teste_2024_2025()
    print(f"Dados reais de teste carregados: {len(dados_teste)} meses (2024 a 2025).")

    # 2. Construir árvore de treino calibrada (2021-2023)
    divisao_8q = [list(range(q * 3, q * 3 + 3)) for q in range(8)]
    arv_treino = construir_arvore_ccee_treino(divisao_estagios=divisao_8q, ramificacao=2, seed=2026)

    # 3. Modelos concorrentes no treino
    print("Treinando modelos na árvore de cenários...")
    m_mer = ModeloMultiestagioTEMPR(arv_treino, _lambda=0.5, permitir_revenda=True)
    m_mer.criar_variaveis()
    m_mer.adicionar_restricoes_posicao_e_balanco()
    m_mer.adicionar_restricoes_revenda()
    m_mer.definir_funcao_objetivo()
    m_mer.resolver(msg=False)

    m_me = ModeloMultiestagioTEMPR(arv_treino, _lambda=0.5, permitir_revenda=False)
    m_me.criar_variaveis()
    m_me.adicionar_restricoes_posicao_e_balanco()
    m_me.adicionar_restricoes_revenda()
    m_me.definir_funcao_objetivo()
    m_me.resolver(msg=False)

    m_heur = ModeloMultiestagioTEMPR(arv_treino, _lambda=0.5, permitir_revenda=False)
    m_heur.criar_variaveis()
    m_heur.adicionar_restricoes_posicao_e_balanco()
    m_heur.fixar_politica_heuristica_mercado(cobertura_pct=0.95)
    m_heur.definir_funcao_objetivo()
    m_heur.resolver(msg=False)

    m_det = resolver_problema_deterministico_cenario_medio(arv_treino, _lambda=0.5)

    # 4. Avaliação fora da amostra (Backtesting 2024-2025)
    print("Avaliando decisões ex-ante no histórico real de 2024-2025...")
    res_mer = simular_politica_no_historico_real(m_mer, dados_teste, "ME-R (Multiestágio com Revenda)")
    res_me = simular_politica_no_historico_real(m_me, dados_teste, "ME (Multiestágio sem Revenda)")
    res_det = simular_politica_no_historico_real(m_det, dados_teste, "Unhedged (DET)")
    res_heur = simular_politica_no_historico_real(m_heur, dados_teste, "HEUR (95% Hedge Passivo)")

    df_backtest = pd.DataFrame([res_mer, res_me, res_det, res_heur])
    
    # 5. Cálculo formal de VSS e EVPI in-sample conforme Birge-Louveaux (teoria clássica com lambda=0.0)
    print("Calculando métricas estocásticas formais (VSS e EVPI)...")
    m_rp0 = ModeloMultiestagioTEMPR(arv_treino, _lambda=0.0, permitir_revenda=True)
    m_rp0.criar_variaveis()
    m_rp0.adicionar_restricoes_posicao_e_balanco()
    m_rp0.adicionar_restricoes_revenda()
    m_rp0.definir_funcao_objetivo()
    m_rp0.resolver(msg=False)
    rp_in_sample = m_rp0.expected_profit / 1e6
    ws_in_sample = resolver_wait_and_see(arv_treino, _lambda=0.0) / 1e6

    m_det0 = resolver_problema_deterministico_cenario_medio(arv_treino, _lambda=0.0)

    m_eev0 = ModeloMultiestagioTEMPR(arv_treino, _lambda=0.0, permitir_revenda=True)
    m_eev0.criar_variaveis()
    m_eev0.adicionar_restricoes_posicao_e_balanco()
    m_eev0.adicionar_restricoes_revenda()
    raiz_id = 0
    for s in m_eev0.SUBMERCADOS:
        def _get_det0(k: str) -> float:
            v = m_det0.variables.get(k)
            return float(v.varValue) if (v is not None and v.varValue is not None) else 0.0

        for n in range(arv_treino.n_months):
            k_p = f"xm_plus[{raiz_id},{s},{n}]"
            k_m = f"xm_minus[{raiz_id},{s},{n}]"
            if k_p in m_eev0.variables:
                m_eev0.lp_model += (m_eev0.variables[k_p] == _get_det0(k_p))
            if k_m in m_eev0.variables:
                m_eev0.lp_model += (m_eev0.variables[k_m] == _get_det0(k_m))
        for q in range(m_eev0.n_quarters):
            k_p = f"xt_plus[{raiz_id},{s},{q}]"
            k_m = f"xt_minus[{raiz_id},{s},{q}]"
            if k_p in m_eev0.variables:
                m_eev0.lp_model += (m_eev0.variables[k_p] == _get_det0(k_p))
            if k_m in m_eev0.variables:
                m_eev0.lp_model += (m_eev0.variables[k_m] == _get_det0(k_m))
        for h in range(m_eev0.n_semesters):
            k_p = f"xs_plus[{raiz_id},{s},{h}]"
            k_m = f"xs_minus[{raiz_id},{s},{h}]"
            if k_p in m_eev0.variables:
                m_eev0.lp_model += (m_eev0.variables[k_p] == _get_det0(k_p))
            if k_m in m_eev0.variables:
                m_eev0.lp_model += (m_eev0.variables[k_m] == _get_det0(k_m))
    m_eev0.definir_funcao_objetivo()
    m_eev0.resolver(msg=False)
    eev_in_sample = m_eev0.expected_profit / 1e6

    # VSS = RP - EEV (maximização de lucro esperado)
    vss = rp_in_sample - eev_in_sample
    # EVPI = WS - RP
    evpi = ws_in_sample - rp_in_sample

    # Validação teórica estrita (Birge & Louveaux 2011)
    assert ws_in_sample >= rp_in_sample - 1e-3, f"Violação WS >= RP: WS={ws_in_sample}, RP={rp_in_sample}"
    assert rp_in_sample >= eev_in_sample - 1e-3, f"Violação RP >= EEV: RP={rp_in_sample}, EEV={eev_in_sample}"

    # Métricas estocásticas sob aversão ao risco (lambda=0.5)
    m_eev_risco = ModeloMultiestagioTEMPR(arv_treino, _lambda=0.5, permitir_revenda=True)
    m_eev_risco.criar_variaveis()
    m_eev_risco.adicionar_restricoes_posicao_e_balanco()
    m_eev_risco.adicionar_restricoes_revenda()
    for s in m_eev_risco.SUBMERCADOS:
        def _get_det_risco(k: str) -> float:
            v = m_det.variables.get(k)
            return float(v.varValue) if (v is not None and v.varValue is not None) else 0.0

        for n in range(arv_treino.n_months):
            k_p = f"xm_plus[{raiz_id},{s},{n}]"
            k_m = f"xm_minus[{raiz_id},{s},{n}]"
            if k_p in m_eev_risco.variables:
                m_eev_risco.lp_model += (m_eev_risco.variables[k_p] == _get_det_risco(k_p))
            if k_m in m_eev_risco.variables:
                m_eev_risco.lp_model += (m_eev_risco.variables[k_m] == _get_det_risco(k_m))
        for q in range(m_eev_risco.n_quarters):
            k_p = f"xt_plus[{raiz_id},{s},{q}]"
            k_m = f"xt_minus[{raiz_id},{s},{q}]"
            if k_p in m_eev_risco.variables:
                m_eev_risco.lp_model += (m_eev_risco.variables[k_p] == _get_det_risco(k_p))
            if k_m in m_eev_risco.variables:
                m_eev_risco.lp_model += (m_eev_risco.variables[k_m] == _get_det_risco(k_m))
        for h in range(m_eev_risco.n_semesters):
            k_p = f"xs_plus[{raiz_id},{s},{h}]"
            k_m = f"xs_minus[{raiz_id},{s},{h}]"
            if k_p in m_eev_risco.variables:
                m_eev_risco.lp_model += (m_eev_risco.variables[k_p] == _get_det_risco(k_p))
            if k_m in m_eev_risco.variables:
                m_eev_risco.lp_model += (m_eev_risco.variables[k_m] == _get_det_risco(k_m))
    m_eev_risco.definir_funcao_objetivo()
    m_eev_risco.resolver(msg=False)

    fo_rp_risco = m_mer.obj_value / 1e6
    fo_me_risco = m_me.obj_value / 1e6
    fo_eev_risco = m_eev_risco.obj_value / 1e6
    vss_fo_risco = fo_rp_risco - fo_eev_risco
    assert fo_rp_risco >= fo_eev_risco - 1e-3, f"Violação FO_RP >= FO_EEV em risco: {fo_rp_risco} vs {fo_eev_risco}"

    # Benchmark Dois Estágios (2S e 2S-R) sob lambda=0.50 (M3: Valor do Recurso Multiestágio)
    m_2s = ModeloMultiestagioTEMPR(arv_treino, _lambda=0.5, permitir_revenda=False)
    m_2s.criar_variaveis()
    m_2s.adicionar_restricoes_posicao_e_balanco()
    m_2s.adicionar_restricoes_revenda()
    m_2s.fixar_politica_dois_estagios()
    m_2s.definir_funcao_objetivo()
    m_2s.resolver(msg=False)

    m_2s_r = ModeloMultiestagioTEMPR(arv_treino, _lambda=0.5, permitir_revenda=True)
    m_2s_r.criar_variaveis()
    m_2s_r.adicionar_restricoes_posicao_e_balanco()
    m_2s_r.adicionar_restricoes_revenda()
    m_2s_r.fixar_politica_dois_estagios()
    m_2s_r.definir_funcao_objetivo()
    m_2s_r.resolver(msg=False)

    fo_2s = m_2s.obj_value / 1e6
    er_2s = m_2s.expected_profit / 1e6
    cvar_profit_2s = -m_2s.cvar_perda / 1e6
    fo_2s_r = m_2s_r.obj_value / 1e6
    vmr_me = fo_me_risco - fo_2s
    vmr_mer = fo_rp_risco - fo_2s

    # Caso sem nenhum hedge em nenhum estágio (q=0 em toda a árvore, lambda=0.50)
    fo_sem_hedge = 0.5 * rp_in_sample + 0.5 * (-m_rp0.cvar_perda / 1e6)

    vss_pct = (vss / eev_in_sample) * 100.0 if eev_in_sample > 0 else 0.0
    evpi_pct = (evpi / rp_in_sample) * 100.0 if rp_in_sample > 0 else 0.0

    metricas_estocasticas = {
        "RP_milhoes": float(rp_in_sample),
        "EEV_milhoes": float(eev_in_sample),
        "WS_milhoes": float(ws_in_sample),
        "VSS_milhoes": float(vss),
        "VSS_pct_sobre_EEV": float(vss_pct),
        "EVPI_milhoes": float(evpi),
        "EVPI_pct_sobre_RP": float(evpi_pct),
        "FO_RP_risco_milhoes": float(fo_rp_risco),
        "FO_ME_risco_milhoes": float(fo_me_risco),
        "FO_EEV_risco_milhoes": float(fo_eev_risco),
        "VSS_FO_risco_milhoes": float(vss_fo_risco),
        "VSS_FO_risco_pct": float((vss_fo_risco / fo_eev_risco) * 100.0) if fo_eev_risco > 0 else 0.0,
        "FO_2S_milhoes": float(fo_2s),
        "ER_2S_milhoes": float(er_2s),
        "CVaR_profit_2S_milhoes": float(cvar_profit_2s),
        "FO_2S_R_milhoes": float(fo_2s_r),
        "VMR_ME_vs_2S_milhoes": float(vmr_me),
        "VMR_MER_vs_2S_milhoes": float(vmr_mer),
        "FO_sem_hedge_milhoes": float(fo_sem_hedge),
    }

    # Salvar resultados
    csv_bt_path = OUTPUT_DIR / "backtesting_historico_real_2024_2025.csv"
    df_backtest.to_csv(csv_bt_path, index=False)
    
    json_metr_path = OUTPUT_DIR / "metricas_estocasticas_vss_evpi.json"
    with open(json_metr_path, "w", encoding="utf-8") as f:
        json.dump(metricas_estocasticas, f, indent=4)

    print("\n=== RESULTADOS DO BACKTESTING FORA DA AMOSTRA (2024-2025) ===")
    print(df_backtest[["politica", "lucro_total_milhoes", "liquidacao_spot_milhoes"]])
    print("\n=== MÉTRICAS CLÁSSICAS DE PROGRAMAÇÃO ESTOCÁSTICA ===")
    print(f"RP (Recourse Problem):        R$ {rp_in_sample:8.2f} Milhões")
    print(f"EEV (Solução do Cenário Médio): R$ {eev_in_sample:8.2f} Milhões")
    print(f"WS (Informação Perfeita):      R$ {ws_in_sample:8.2f} Milhões")
    print(f"VSS (Valor da Solução Estocástica): R$ {vss:8.2f} Milhões (+{metricas_estocasticas['VSS_pct_sobre_EEV']:.2f}%)")
    print(f"EVPI (Valor da Informação Perfeita): R$ {evpi:8.2f} Milhões (+{metricas_estocasticas['EVPI_pct_sobre_RP']:.2f}%)")
    print(f"2S (Dois Estágios, lambda=0.5):  FO=R$ {fo_2s:7.2f}M (E[R]={er_2s:.2f}M, CVaR5%={cvar_profit_2s:.2f}M)")
    print(f"VMR (ME - 2S):                   +R$ {vmr_me:.2f}M | (ME-R - 2S): +R$ {vmr_mer:.2f}M")
    print(f"EEV risco (q0=0, recurso t>=1):  FO=R$ {fo_eev_risco:7.2f}M | Sem hedge total: R$ {fo_sem_hedge:.2f}M")

    return df_backtest, metricas_estocasticas


if __name__ == "__main__":
    executar_backtesting_e_metricas()
