"""
Modelo Multiestágio de Posição Líquida e Risco para o Artigo TEMPR.
(IEEE Transactions on Energy Markets, Policy and Regulation)

Revisão Metodológica Completa:
1. Receita de Venda a Clientes (P_venda * D * horas_mes):
   - Preço de venda no ACL a clientes finais (default: R$ 160,00/MWh, margem realista).
   - Conversão estrita MWmédio -> MWh multiplicando pelas horas do mês (horas_mes).
   - Resultados e fluxos expressos em R$ reais (e reportados em Milhões de R$).
2. Preços a Termo Ex-Ante:
   - Neutro ao risco (expectativa condicional no nó raiz/nó) ou com prêmio de risco paramétrico.
   - Sem atrelar artificialmente ao cluster local de cada nó.
3. Incerteza de Demanda por Nó:
   - Demanda efetiva no nó nu: D[nu, s, m] = D_base[s, m] * demanda_fator[nu, m].
   - Captura churn, perda de clientes ou choques de carga, ativando o valor econômico da revenda.
4. Preços Spot Regionais CCEE:
   - Usa pld_se e pld_ne reais por nó gerados a partir do histórico CCEE.
   - Permite contratos num submercado e carga no outro, liquidando o diferencial de submercado.
5. Revenda Limitada ao Volume Já Contratado:
   - Vendas acumuladas <= Compras acumuladas + Geração física própria + Posição inicial (sem folga artificial).
6. Limites de PLD por Ano Calendário (Resoluções ANEEL):
   - 2021: [piso: 49.77, teto estrutural: 583.88, teto horário: 1197.87] (REH 2.828/2020)
   - 2022: [piso: 55.70, teto estrutural: 646.58, teto horário: 1326.50] (REH 2.994/2021 + Despacho SGT 4.046/2021)
   - 2023: [piso: 69.04, teto estrutural: 684.73, teto horário: 1404.77] (REH 3.167/2022 retificada)
   - 2024: [piso: 61.07, teto estrutural: 716.80, teto horário: 1470.57] (REH 3.304/2023)
   - 2025: [piso: 58.60, teto estrutural: 751.73, teto horário: 1542.23] (Despacho 3.625/2024)
7. Encargos Setoriais (ESS/EER) e Lastro (Decreto 5.163/2004):
   - Encargos incidem como custo regulatório fixo sobre a demanda física (R$ 15,00/MWh * D * horas).
   - Penalidade opcional por insuficiência de lastro (VR CCEE atualizado para 2025 = R$ 231,47/MWh,
     correspondente ao VR de R$ 189,90/MWh base jul/2021 do Despacho ANEEL 3.176/2024 atualizado pelo IPCA,
     nos termos da REN ANEEL 1.090/2024) em janela móvel de 12 meses e piso mínimo opcional de cobertura ex-ante.
8. Heurística de Mercado:
   - Contratação passiva ex-ante de 95% do consumo no nó raiz, com contratos mensais.
9. CVaR Terminal Canônico Pós-Solução:
   - Avaliado de forma consistente para qualquer lambda.
10. Neutralidade ao Risco e Contratos a Termo:
   - Com preço justo condicional (martingale) e spread > 0, compras a termo têm retorno esperado estritamente negativo.
   - Sob lambda = 0 (neutralidade) e sem penalidade/piso de lastro, a carteira ótima é zero contratos a termo (ME-R = ME).
   - Contratos a termo só são contratados com lambda > 0 ou sob exigência regulatória de lastro.
11. Restrição Física de Posição:
   - 0 <= q[nu, s, n] <= kappa * D_max[s, n] (kappa ~= 1.0 - 1.2), eliminando limites artificiais ativos.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pulp

from arvore_cenarios import ArvoreCenarios, NoCenario


class ModeloMultiestagioTEMPR:
    """Modelo estocástico multiestágio de comercialização de energia adaptado às regras CCEE/TEMPR."""

    SUBMERCADOS = ["SE", "NE"]

    # Horas padrão dos 24 meses (ano comum: 2021 e 2022)
    HORAS_MES_PADRAO = np.array([
        744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744,  # Ano 1 (2021)
        744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744   # Ano 2 (2022)
    ], dtype=float)

    # Limites regulatórios ANEEL por ano [piso, teto estrutural (médias mensais), teto horário]
    LIMITES_ANEEL = {
        2021: {"piso": 49.77, "teto_estrutural": 583.88, "teto_horario": 1197.87},
        2022: {"piso": 55.70, "teto_estrutural": 646.58, "teto_horario": 1326.50},
        2023: {"piso": 69.04, "teto_estrutural": 684.73, "teto_horario": 1404.77},
        2024: {"piso": 61.07, "teto_estrutural": 716.80, "teto_horario": 1470.57},
        2025: {"piso": 58.60, "teto_estrutural": 751.73, "teto_horario": 1542.23},
    }

    def __init__(
        self,
        arvore: ArvoreCenarios,
        demanda_se: Optional[np.ndarray] = None,
        demanda_ne: Optional[np.ndarray] = None,
        geracao_se: Optional[np.ndarray] = None,
        geracao_ne: Optional[np.ndarray] = None,
        preco_venda_cliente_mwh: float = 160.0,
        horas_mes: Optional[np.ndarray] = None,
        _lambda: float = 0.5,
        beta: float = 0.95,
        permitir_revenda: bool = True,
        encargo_ccee_mwh: float = 15.00,
        ano_inicio: int = 2021,
        solver_name: str = "HiGHS",
        limite_posicao_mw: float = 250.0,
        spread_termo: float = 0.01,
        premio_risco_termo: float = 0.0,
        decaimento_premio: bool = False,
        kappa_demanda_max: float = 1.0,
        nome_modelo: str = "TEMPR_Multiestagio",
        posicao_inicial: Optional[Dict[str, np.ndarray]] = None,
        mes_inicio_calendario: int = 0,
        fator_captura_demanda: Optional[Dict[str, np.ndarray]] = None,
        permitir_contrato_cruzado: bool = False,
        custo_ftr_mwh: Optional[float] = None,
        penalidade_lastro_mwh: float = 0.0,
        cobertura_minima_lastro: float = 0.0,
        lastro_janela_movel: bool = True,
    ):
        self.arvore = arvore
        self.n_months = arvore.n_months
        self._lambda = float(_lambda)
        self.beta = float(beta)
        self.permitir_revenda = permitir_revenda
        self.preco_venda_cliente_mwh = float(preco_venda_cliente_mwh)
        self.encargo_ccee_mwh = float(encargo_ccee_mwh)
        self.ano_inicio = int(ano_inicio)
        self.mes_inicio_calendario = int(mes_inicio_calendario)
        self.solver_name = solver_name
        self.limite_posicao_mw = float(limite_posicao_mw) if limite_posicao_mw is not None else 1000.0
        self.spread_termo = float(spread_termo)
        self.premio_risco_termo = float(premio_risco_termo)
        self.decaimento_premio = bool(decaimento_premio)
        self.kappa_demanda_max = float(kappa_demanda_max)
        self.nome_modelo = nome_modelo
        self.permitir_contrato_cruzado = bool(permitir_contrato_cruzado)
        self.custo_ftr_mwh = float(custo_ftr_mwh) if custo_ftr_mwh is not None else None
        self.penalidade_lastro_mwh = float(penalidade_lastro_mwh)
        self.cobertura_minima_lastro = float(cobertura_minima_lastro)
        self.lastro_janela_movel = bool(lastro_janela_movel)
        self.fator_captura_demanda = (
            {
                s: np.array(fator_captura_demanda[s], dtype=float)
                if s in fator_captura_demanda
                else np.ones(self.n_months, dtype=float)
                for s in self.SUBMERCADOS
            }
            if fator_captura_demanda is not None
            else None
        )
        self.posicao_inicial = (
            {s: np.array(posicao_inicial[s], dtype=float) for s in self.SUBMERCADOS if s in posicao_inicial}
            if posicao_inicial is not None
            else None
        )

        # Horas de cada mês
        if horas_mes is not None:
            self.horas_mes = np.array(horas_mes, dtype=float)
        elif len(self.HORAS_MES_PADRAO) == self.n_months:
            self.horas_mes = self.HORAS_MES_PADRAO.copy()
        else:
            self.horas_mes = np.full(self.n_months, 720.0, dtype=float)

        # Demandas base em MWmédio (default: 50 MWm SE, 20 MWm NE)
        self.D_base = {
            "SE": np.full(self.n_months, 50.0, dtype=float) if demanda_se is None else np.array(demanda_se, dtype=float),
            "NE": np.full(self.n_months, 20.0, dtype=float) if demanda_ne is None else np.array(demanda_ne, dtype=float),
        }

        # Geração física própria em MWmédio
        self.E_base = {
            "SE": np.zeros(self.n_months, dtype=float) if geracao_se is None else np.array(geracao_se, dtype=float),
            "NE": np.zeros(self.n_months, dtype=float) if geracao_ne is None else np.array(geracao_ne, dtype=float),
        }

        # Demanda máxima possível por submercado e mês (considerando incerteza/demanda_fator por nó)
        self.D_max = {
            "SE": np.zeros(self.n_months, dtype=float),
            "NE": np.zeros(self.n_months, dtype=float),
        }
        for s in self.SUBMERCADOS:
            for n in range(self.n_months):
                self.D_max[s][n] = max(
                    (float(self._obter_demanda_no(node, s)[n]) for node in self.arvore.nodes.values()),
                    default=float(self.D_base[s][n]),
                )

        self.lp_model = pulp.LpProblem(self.nome_modelo, pulp.LpMaximize)
        self.variables: Dict[str, pulp.LpVariable] = {}

        self.n_quarters = self.n_months // 3
        self.n_semesters = self.n_months // 6

        # Variáveis de cauda do CVaR
        self.alfa = pulp.LpVariable("alfa", cat="Continuous")
        self.u_vars: Dict[int, pulp.LpVariable] = {}

        # Armazenamento pós-solução
        self.expected_profit: Optional[float] = None
        self.cvar_perda: Optional[float] = None
        self.var_perda_alfa: Optional[float] = None
        self.obj_value: Optional[float] = None
        self.lucro_folhas: Dict[int, float] = {}
        self.posicao_nos: Dict[Tuple[int, str, int], float] = {}
        self.fluxo_nos: Dict[int, float] = {}

        # Pré-computar preços a termo livres de arbitragem por nó: P_fwd[nu, s, n] = E[PLD_{s,n} | nu] * (1 +- spread)
        self._preparar_precos_termo_por_no()

    def _obter_limites_ano(self, mes_idx: int) -> Tuple[float, float]:
        """Retorna (piso, teto_estrutural) da ANEEL para o ano civil do mês."""
        ano = self.ano_inicio + ((self.mes_inicio_calendario + mes_idx) // 12)
        lim = self.LIMITES_ANEEL.get(ano, {"piso": 49.77, "teto_estrutural": 751.73, "teto_horario": 1542.23})
        return float(lim["piso"]), float(lim["teto_estrutural"])

    def _obter_pld_no(self, node: NoCenario, submercado: str) -> np.ndarray:
        """Obtém o vetor de PLD spot do nó aplicando limites regulatórios da ANEEL por ano civil."""
        if submercado == "SE" and node.pld_se is not None and len(node.pld_se) == self.n_months:
            pld_raw = node.pld_se.copy()
        elif submercado == "NE" and node.pld_ne is not None and len(node.pld_ne) == self.n_months:
            pld_raw = node.pld_ne.copy()
        elif hasattr(node, "pld_submercados") and node.pld_submercados and submercado in node.pld_submercados:
            pld_raw = node.pld_submercados[submercado].copy()
        else:
            pld_raw = node.pld.copy() if len(node.pld) == self.n_months else np.full(self.n_months, 100.0)

        # Aplicar limites regulatórios ano a ano (piso e teto estrutural na média mensal)
        pld_clipped = np.zeros(self.n_months, dtype=float)
        for m in range(self.n_months):
            pmin, pmax = self._obter_limites_ano(m)
            pld_clipped[m] = np.clip(pld_raw[m], pmin, pmax)
        return pld_clipped

    def _obter_demanda_no(self, node: NoCenario, submercado: str) -> np.ndarray:
        """Obtém a demanda efetiva em MWmédio no nó (aplicando incerteza/demanda_fator)."""
        fator = node.demanda_fator if (node.demanda_fator is not None and len(node.demanda_fator) == self.n_months) else np.ones(self.n_months)
        return self.D_base[submercado] * fator

    def _obter_folhas_descendentes(self, node: NoCenario) -> List[NoCenario]:
        """Retorna lista de folhas que descendem do nó (ou a própria folha)."""
        if node.is_leaf:
            return [node]
        folhas = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.is_leaf:
                folhas.append(cur)
            else:
                for cid in cur.children_ids:
                    stack.append(self.arvore.nodes[cid])
        return folhas

    def _preparar_precos_termo_por_no(self):
        """
        Calibra preços a termo POR NÓ, eliminando arbitragem intertemporal:
        P_fwd[nu, s, n] = E[PLD_{s,n} | nu] * (1 +- spread).
        E[PLD_{s,n} | nu] é a média ponderada das folhas descendentes de nu:
        w_l = prob(l) / prob(nu).
        """
        self.precos_termo: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

        for nid, node in self.arvore.nodes.items():
            folhas = self._obter_folhas_descendentes(node)
            prob_nu = node.prob if node.prob > 0 else 1.0
            pesos = np.array([f.prob / prob_nu for f in folhas], dtype=float)
            soma_pesos = float(np.sum(pesos))
            if abs(soma_pesos - 1.0) > 1e-5 and soma_pesos > 0:
                pesos /= soma_pesos

            self.precos_termo[nid] = {}

            plds_se_folhas = np.array([self._obter_pld_no(f, "SE") for f in folhas], dtype=float)
            plds_ne_folhas = np.array([self._obter_pld_no(f, "NE") for f in folhas], dtype=float)
            spread_pos_folhas = np.maximum(0.0, plds_se_folhas - plds_ne_folhas)
            self.precos_termo[nid]["_ftr_esp"] = np.dot(pesos, spread_pos_folhas)

            m_ref = max(node.meses) if node.meses else node.mes_decisao
            denom_tau = max(1.0, float(self.n_months - 3))

            for s in self.SUBMERCADOS:
                # Matriz (n_folhas, n_months) com PLD das folhas
                plds_folhas = plds_se_folhas if s == "SE" else plds_ne_folhas
                # E[PLD_{s,n} | nu]
                pld_esp_no = np.dot(pesos, plds_folhas)

                # Preços a termo por nó com bid-ask spread e prêmio de risco opcional (constante ou decaindo com o prazo)
                if self.decaimento_premio and abs(self.premio_risco_termo) > 0.0:
                    taus_m = np.maximum(0.0, np.arange(self.n_months, dtype=float) - m_ref) / denom_tau
                    fator_m = 1.0 + self.premio_risco_termo * taus_m
                else:
                    fator_m = 1.0 + self.premio_risco_termo

                # Preços mensais
                pm_v = pld_esp_no * fator_m * (1.0 - self.spread_termo)
                pm_c = pld_esp_no * fator_m * (1.0 + self.spread_termo)

                # Preços trimestrais
                pt_v = np.zeros(self.n_quarters, dtype=float)
                pt_c = np.zeros(self.n_quarters, dtype=float)
                for q in range(self.n_quarters):
                    start = q * 3
                    end = start + 3
                    h_q = self.horas_mes[start:end]
                    base_q = float(np.sum(pld_esp_no[start:end] * h_q) / np.sum(h_q))
                    if self.decaimento_premio and abs(self.premio_risco_termo) > 0.0:
                        tau_q = max(0.0, (start + 1.0) - m_ref) / denom_tau
                        fator_q = 1.0 + self.premio_risco_termo * tau_q
                    else:
                        fator_q = 1.0 + self.premio_risco_termo
                    pt_v[q] = base_q * fator_q * (1.0 - self.spread_termo * 0.8)
                    pt_c[q] = base_q * fator_q * (1.0 + self.spread_termo * 0.8)

                # Preços semestrais
                ps_v = np.zeros(self.n_semesters, dtype=float)
                ps_c = np.zeros(self.n_semesters, dtype=float)
                for h in range(self.n_semesters):
                    start = h * 6
                    end = start + 6
                    h_h = self.horas_mes[start:end]
                    base_h = float(np.sum(pld_esp_no[start:end] * h_h) / np.sum(h_h))
                    if self.decaimento_premio and abs(self.premio_risco_termo) > 0.0:
                        tau_h = max(0.0, (start + 2.5) - m_ref) / denom_tau
                        fator_h = 1.0 + self.premio_risco_termo * tau_h
                    else:
                        fator_h = 1.0 + self.premio_risco_termo
                    ps_v[h] = base_h * fator_h * (1.0 - self.spread_termo * 0.6)
                    ps_c[h] = base_h * fator_h * (1.0 + self.spread_termo * 0.6)

                self.precos_termo[nid][s] = {
                    "pld_esp": pld_esp_no,
                    "pm_v": pm_v, "pm_c": pm_c,
                    "pt_v": pt_v, "pt_c": pt_c,
                    "ps_v": ps_v, "ps_c": ps_c,
                }

    # Alias para compatibilidade
    def _preparar_precos_termo_ex_ante(self):
        self._preparar_precos_termo_por_no()

    def criar_variaveis(self):
        """
        Cria variáveis de decisão em MWmédio.
        Regra geral de produtos futuros:
        Um produto a termo (mensal n, trimestral q, semestral h) só pode ser negociado no nó nu
        se NENHUM de seus meses de entrega pertencer aos meses já revelados no nó:
        min(meses_produto) >= max(node.meses) + 1.
        Isso proíbe qualquer contratação intraestágio ou de meses passados, qualquer que seja
        a partição temporal dos estágios (trimestrais, mensais ou irregulares).

        Restrição física de posição líquida:
        0 <= q[nu, s, n] <= kappa * D_max[s, n] (kappa ~= 1.0 - 1.2).
        Elimina o limite artificial limite_posicao_mw como restrição ativa: a carteira
        é dimensionada pela escala física da demanda dos clientes da comercializadora.
        """
        for nid, node in self.arvore.nodes.items():
            t = node.mes_decisao
            n_min_futuro = (max(node.meses) + 1) if node.meses else (t + 1)

            for s in self.SUBMERCADOS:
                # 1. Contratos mensais: apenas para meses futuros n >= n_min_futuro
                for n in range(n_min_futuro, self.n_months):
                    self.variables[f"xm_plus[{nid},{s},{n}]"] = pulp.LpVariable(
                        f"xm_p_{nid}_{s}_{n}", lowBound=0, cat="Continuous"
                    )
                    self.variables[f"xm_minus[{nid},{s},{n}]"] = pulp.LpVariable(
                        f"xm_m_{nid}_{s}_{n}", lowBound=0, cat="Continuous"
                    )

                # 2. Contratos trimestrais: apenas para trimestres futuros cujo início 3*q >= n_min_futuro
                for q in range(self.n_quarters):
                    if q * 3 >= n_min_futuro:
                        self.variables[f"xt_plus[{nid},{s},{q}]"] = pulp.LpVariable(
                            f"xt_p_{nid}_{s}_{q}", lowBound=0, cat="Continuous"
                        )
                        self.variables[f"xt_minus[{nid},{s},{q}]"] = pulp.LpVariable(
                            f"xt_m_{nid}_{s}_{q}", lowBound=0, cat="Continuous"
                        )

                # 3. Contratos semestrais: apenas para semestres futuros cujo início 6*h >= n_min_futuro
                for h in range(self.n_semesters):
                    if h * 6 >= n_min_futuro:
                        self.variables[f"xs_plus[{nid},{s},{h}]"] = pulp.LpVariable(
                            f"xs_p_{nid}_{s}_{h}", lowBound=0, cat="Continuous"
                        )
                        self.variables[f"xs_minus[{nid},{s},{h}]"] = pulp.LpVariable(
                            f"xs_m_{nid}_{s}_{h}", lowBound=0, cat="Continuous"
                        )

                # 4. Posição contratada acumulada q[nu, s, n]: definida para n >= t
                # Restrição física estrita: 0 <= q[nu, s, n] <= kappa * D_max[s, n]
                for n in range(t, self.n_months):
                    cap_fisico = self.kappa_demanda_max * float(self.D_max[s][n])
                    if self.posicao_inicial is not None and s in self.posicao_inicial and n < len(self.posicao_inicial[s]):
                        cap_fisico = max(cap_fisico, float(self.posicao_inicial[s][n]))
                    teto_q = min(self.limite_posicao_mw, cap_fisico) if self.limite_posicao_mw is not None else cap_fisico
                    self.variables[f"q[{nid},{s},{n}]"] = pulp.LpVariable(
                        f"q_{nid}_{s}_{n}",
                        lowBound=0.0,
                        upBound=teto_q,
                        cat="Continuous",
                    )

                # 5. Liquidação física spot (sobras e déficits em MWmédio)
                for m in node.meses:
                    self.variables[f"y_plus[{nid},{s},{m}]"] = pulp.LpVariable(
                        f"y_p_{nid}_{s}_{m}", lowBound=0, cat="Continuous"
                    )
                    self.variables[f"y_minus[{nid},{s},{m}]"] = pulp.LpVariable(
                        f"y_m_{nid}_{s}_{m}", lowBound=0, cat="Continuous"
                    )

            # 6. Contratos cruzados opcionais NE->SE (compra no NE para cobrir carga SE)
            if self.permitir_contrato_cruzado:
                for n in range(n_min_futuro, self.n_months):
                    self.variables[f"xmc_plus[{nid},{n}]"] = pulp.LpVariable(
                        f"xmc_p_{nid}_{n}", lowBound=0, cat="Continuous"
                    )
                    self.variables[f"xmc_minus[{nid},{n}]"] = pulp.LpVariable(
                        f"xmc_m_{nid}_{n}", lowBound=0, cat="Continuous"
                    )
                    if self.custo_ftr_mwh is not None:
                        self.variables[f"ftr_plus[{nid},{n}]"] = pulp.LpVariable(
                            f"ftr_p_{nid}_{n}", lowBound=0, cat="Continuous"
                        )
                for q in range(self.n_quarters):
                    if q * 3 >= n_min_futuro:
                        self.variables[f"xtc_plus[{nid},{q}]"] = pulp.LpVariable(
                            f"xtc_p_{nid}_{q}", lowBound=0, cat="Continuous"
                        )
                        self.variables[f"xtc_minus[{nid},{q}]"] = pulp.LpVariable(
                            f"xtc_m_{nid}_{q}", lowBound=0, cat="Continuous"
                        )
                for h in range(self.n_semesters):
                    if h * 6 >= n_min_futuro:
                        self.variables[f"xsc_plus[{nid},{h}]"] = pulp.LpVariable(
                            f"xsc_p_{nid}_{h}", lowBound=0, cat="Continuous"
                        )
                        self.variables[f"xsc_minus[{nid},{h}]"] = pulp.LpVariable(
                            f"xsc_m_{nid}_{h}", lowBound=0, cat="Continuous"
                        )
                for n in range(t, self.n_months):
                    cap_se = self.kappa_demanda_max * float(self.D_max["SE"][n])
                    self.variables[f"qc[{nid},{n}]"] = pulp.LpVariable(
                        f"qc_{nid}_{n}", lowBound=0.0, upBound=cap_se, cat="Continuous"
                    )
                    if self.custo_ftr_mwh is not None:
                        self.variables[f"qftr[{nid},{n}]"] = pulp.LpVariable(
                            f"qftr_{nid}_{n}", lowBound=0.0, upBound=cap_se, cat="Continuous"
                        )

        # Variáveis de cauda do CVaR nas folhas
        for leaf in self.arvore.leaves:
            self.u_vars[leaf.id] = pulp.LpVariable(
                f"u_{leaf.id}", lowBound=0, cat="Continuous"
            )

        # Variáveis opcionais de insuficiência de lastro (Decreto 5.163/2004, VR = R$ 189,90/MWh)
        self.deficit_lastro_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
        if self.penalidade_lastro_mwh > 0.0:
            raiz = self.arvore.root
            n_min_raiz = (max(raiz.meses) + 1) if raiz.meses else 1
            m_audit_start = (n_min_raiz + 11) if (self.lastro_janela_movel and self.n_months >= n_min_raiz + 12) else n_min_raiz
            for leaf in self.arvore.leaves:
                for m in range(m_audit_start, self.n_months):
                    self.deficit_lastro_vars[(leaf.id, m)] = pulp.LpVariable(
                        f"def_lastro_{leaf.id}_{m}", lowBound=0.0, cat="Continuous"
                    )

    def adicionar_restricoes_posicao_e_balanco(self):
        """Dinâmica de não-antecipatividade, piso mínimo de cobertura e balanço físico por submercado."""
        raiz = self.arvore.root
        n_min_futuro_raiz = (max(raiz.meses) + 1) if raiz.meses else 1

        for nid, node in self.arvore.nodes.items():
            t = node.mes_decisao
            pai_id = node.parent_id

            for s in self.SUBMERCADOS:
                # Dinâmica da posição líquida q[nu, s, n] para n >= t
                for n in range(t, self.n_months):
                    pos_atual = self.variables[f"q[{nid},{s},{n}]"]
                    pos_anterior = (
                        (
                            float(self.posicao_inicial[s][n])
                            if (self.posicao_inicial is not None and s in self.posicao_inicial and n < len(self.posicao_inicial[s]))
                            else 0.0
                        )
                        if pai_id is None
                        else self.variables[f"q[{pai_id},{s},{n}]"]
                    )

                    delta_m = 0
                    if f"xm_plus[{nid},{s},{n}]" in self.variables:
                        delta_m = (
                            self.variables[f"xm_plus[{nid},{s},{n}]"]
                            - self.variables[f"xm_minus[{nid},{s},{n}]"]
                        )

                    q_idx = n // 3
                    delta_t = 0
                    if f"xt_plus[{nid},{s},{q_idx}]" in self.variables:
                        delta_t = (
                            self.variables[f"xt_plus[{nid},{s},{q_idx}]"]
                            - self.variables[f"xt_minus[{nid},{s},{q_idx}]"]
                        )

                    h_idx = n // 6
                    delta_s = 0
                    if f"xs_plus[{nid},{s},{h_idx}]" in self.variables:
                        delta_s = (
                            self.variables[f"xs_plus[{nid},{s},{h_idx}]"]
                            - self.variables[f"xs_minus[{nid},{s},{h_idx}]"]
                        )

                    self.lp_model += (
                        pos_atual == pos_anterior + delta_m + delta_t + delta_s,
                        f"pos_dinamica_{nid}_{s}_{n}",
                    )

                    # Piso regulatório ex-ante de cobertura contratual (para meses contratáveis n >= n_min_futuro_raiz)
                    if self.cobertura_minima_lastro > 0.0 and n >= n_min_futuro_raiz:
                        e_val_n = float(self.E_base[s][n])
                        piso_mw = self.cobertura_minima_lastro * float(self.D_base[s][n])
                        self.lp_model += (
                            pos_atual + e_val_n >= piso_mw,
                            f"piso_lastro_{nid}_{s}_{n}",
                        )

                # Balanço físico: y_plus - y_minus = E - D_efetivo + q (+ qc no NE se contrato cruzado ativo)
                d_efetivo = self._obter_demanda_no(node, s)
                for m in node.meses:
                    d_val = float(d_efetivo[m])
                    e_val = float(self.E_base[s][m])
                    q_val = self.variables[f"q[{nid},{s},{m}]"]
                    y_p = self.variables[f"y_plus[{nid},{s},{m}]"]
                    y_m = self.variables[f"y_minus[{nid},{s},{m}]"]
                    qc_extra = (
                        self.variables[f"qc[{nid},{m}]"]
                        if (self.permitir_contrato_cruzado and s == "NE")
                        else 0.0
                    )

                    self.lp_model += (
                        y_p - y_m == e_val - d_val + q_val + qc_extra,
                        f"balanco_{nid}_{s}_{m}",
                    )

            # Dinâmica e teto físico do contrato cruzado NE->SE
            if self.permitir_contrato_cruzado:
                for n in range(t, self.n_months):
                    qc_atual = self.variables[f"qc[{nid},{n}]"]
                    qc_ant = 0.0 if pai_id is None else self.variables[f"qc[{pai_id},{n}]"]

                    dm_c = 0
                    if f"xmc_plus[{nid},{n}]" in self.variables:
                        dm_c = self.variables[f"xmc_plus[{nid},{n}]"] - self.variables[f"xmc_minus[{nid},{n}]"]
                    q_idx = n // 3
                    dt_c = 0
                    if f"xtc_plus[{nid},{q_idx}]" in self.variables:
                        dt_c = self.variables[f"xtc_plus[{nid},{q_idx}]"] - self.variables[f"xtc_minus[{nid},{q_idx}]"]
                    h_idx = n // 6
                    ds_c = 0
                    if f"xsc_plus[{nid},{h_idx}]" in self.variables:
                        ds_c = self.variables[f"xsc_plus[{nid},{h_idx}]"] - self.variables[f"xsc_minus[{nid},{h_idx}]"]

                    self.lp_model += (
                        qc_atual == qc_ant + dm_c + dt_c + ds_c,
                        f"qc_dinamica_{nid}_{n}",
                    )
                    cap_se = self.kappa_demanda_max * float(self.D_max["SE"][n])
                    self.lp_model += (
                        self.variables[f"q[{nid},SE,{n}]"] + qc_atual <= cap_se,
                        f"cap_combinado_se_cross_{nid}_{n}",
                    )
                    if self.custo_ftr_mwh is not None:
                        qftr_atual = self.variables[f"qftr[{nid},{n}]"]
                        qftr_ant = 0.0 if pai_id is None else self.variables[f"qftr[{pai_id},{n}]"]
                        dftr = self.variables[f"ftr_plus[{nid},{n}]"] if f"ftr_plus[{nid},{n}]" in self.variables else 0.0
                        self.lp_model += (qftr_atual == qftr_ant + dftr, f"qftr_dinamica_{nid}_{n}")
                        self.lp_model += (qftr_atual <= qc_atual, f"qftr_cap_{nid}_{n}")

    def adicionar_restricoes_revenda(self):
        """
        Regra Estrita de Revenda:
        - Vendas acumuladas <= Compras acumuladas + Geração física própria + Posição inicial (sem folga artificial).
        - Sem revenda: Vendas acumuladas <= Geração física própria.
        """
        for nid, node in self.arvore.nodes.items():
            caminho = self.arvore.caminho_ate_raiz(nid)
            t = node.mes_decisao

            for s in self.SUBMERCADOS:
                for n in range(t, self.n_months):
                    vendas_cum = []
                    compras_cum = []
                    for anc in caminho:
                        if f"xm_minus[{anc},{s},{n}]" in self.variables:
                            vendas_cum.append(self.variables[f"xm_minus[{anc},{s},{n}]"])
                        if f"xm_plus[{anc},{s},{n}]" in self.variables:
                            compras_cum.append(self.variables[f"xm_plus[{anc},{s},{n}]"])
                        q_idx = n // 3
                        if f"xt_minus[{anc},{s},{q_idx}]" in self.variables:
                            vendas_cum.append(self.variables[f"xt_minus[{anc},{s},{q_idx}]"])
                        if f"xt_plus[{anc},{s},{q_idx}]" in self.variables:
                            compras_cum.append(self.variables[f"xt_plus[{anc},{s},{q_idx}]"])
                        h_idx = n // 6
                        if f"xs_minus[{anc},{s},{h_idx}]" in self.variables:
                            vendas_cum.append(self.variables[f"xs_minus[{anc},{s},{h_idx}]"])
                        if f"xs_plus[{anc},{s},{h_idx}]" in self.variables:
                            compras_cum.append(self.variables[f"xs_plus[{anc},{s},{h_idx}]"])

                    e_val = float(self.E_base[s][n])
                    q_init = (
                        float(self.posicao_inicial[s][n])
                        if (self.posicao_inicial is not None and s in self.posicao_inicial and n < len(self.posicao_inicial[s]))
                        else 0.0
                    )
                    if not self.permitir_revenda:
                        if vendas_cum:
                            self.lp_model += (
                                pulp.lpSum(vendas_cum) <= e_val,
                                f"sem_revenda_{nid}_{s}_{n}",
                            )
                    else:
                        if vendas_cum:
                            self.lp_model += (
                                pulp.lpSum(vendas_cum) <= pulp.lpSum(compras_cum) + e_val + q_init,
                                f"revenda_estrita_{nid}_{s}_{n}",
                            )

            if self.permitir_contrato_cruzado:
                for n in range(t, self.n_months):
                    vc_cum, cc_cum = [], []
                    for anc in caminho:
                        if f"xmc_minus[{anc},{n}]" in self.variables:
                            vc_cum.append(self.variables[f"xmc_minus[{anc},{n}]"])
                        if f"xmc_plus[{anc},{n}]" in self.variables:
                            cc_cum.append(self.variables[f"xmc_plus[{anc},{n}]"])
                        q_idx = n // 3
                        if f"xtc_minus[{anc},{q_idx}]" in self.variables:
                            vc_cum.append(self.variables[f"xtc_minus[{anc},{q_idx}]"])
                        if f"xtc_plus[{anc},{q_idx}]" in self.variables:
                            cc_cum.append(self.variables[f"xtc_plus[{anc},{q_idx}]"])
                        h_idx = n // 6
                        if f"xsc_minus[{anc},{h_idx}]" in self.variables:
                            vc_cum.append(self.variables[f"xsc_minus[{anc},{h_idx}]"])
                        if f"xsc_plus[{anc},{h_idx}]" in self.variables:
                            cc_cum.append(self.variables[f"xsc_plus[{anc},{h_idx}]"])
                    if vc_cum:
                        if not self.permitir_revenda:
                            self.lp_model += (pulp.lpSum(vc_cum) <= 0.0, f"sem_revenda_cross_{nid}_{n}")
                        else:
                            self.lp_model += (
                                pulp.lpSum(vc_cum) <= pulp.lpSum(cc_cum),
                                f"revenda_estrita_cross_{nid}_{n}",
                            )

    def fixar_politica_dois_estagios(self):
        """
        Benchmark Dois Estágios (2S) sob a mesma árvore e aversão ao risco:
        - Todas as decisões de contratação/revenda a termo são tomadas exclusivamente na raiz (t = 0).
        - Todos os estágios subsequentes (nid != 0) ficam proibidos de negociar novos contratos ou revender,
          mantendo apenas o recurso físico de liquidação spot (y_plus, y_minus).
        """
        raiz_id = 0
        for nid in self.arvore.nodes:
            if nid == raiz_id:
                continue
            for s in self.SUBMERCADOS:
                for n in range(self.n_months):
                    if f"xm_plus[{nid},{s},{n}]" in self.variables:
                        self.lp_model += (self.variables[f"xm_plus[{nid},{s},{n}]"] == 0.0, f"twostage_no_buy_{nid}_{s}_{n}")
                        self.lp_model += (self.variables[f"xm_minus[{nid},{s},{n}]"] == 0.0, f"twostage_no_sell_{nid}_{s}_{n}")
                for q in range(self.n_quarters):
                    if f"xt_plus[{nid},{s},{q}]" in self.variables:
                        self.lp_model += (self.variables[f"xt_plus[{nid},{s},{q}]"] == 0.0, f"twostage_no_xt_p_{nid}_{s}_{q}")
                        self.lp_model += (self.variables[f"xt_minus[{nid},{s},{q}]"] == 0.0, f"twostage_no_xt_m_{nid}_{s}_{q}")
                for h in range(self.n_semesters):
                    if f"xs_plus[{nid},{s},{h}]" in self.variables:
                        self.lp_model += (self.variables[f"xs_plus[{nid},{s},{h}]"] == 0.0, f"twostage_no_xs_p_{nid}_{s}_{h}")
                        self.lp_model += (self.variables[f"xs_minus[{nid},{s},{h}]"] == 0.0, f"twostage_no_xs_m_{nid}_{s}_{h}")

    def fixar_politica_heuristica_mercado(self, cobertura_pct: float = 0.95):
        """
        Heurística de mercado tradicional:
        - Contrata ex-ante no nó raiz cobertura_pct da demanda média de cada submercado
          para todos os meses futuros permitidos (n >= n_min_futuro).
        - Sem recontratação nem revenda nos estágios futuros.
        """
        raiz_id = 0
        raiz = self.arvore.root
        n_min_futuro = (max(raiz.meses) + 1) if raiz.meses else 1

        for s in self.SUBMERCADOS:
            for n in range(n_min_futuro, self.n_months):
                if f"xm_plus[{raiz_id},{s},{n}]" in self.variables:
                    target_volume = cobertura_pct * float(self.D_base[s][n])
                    self.lp_model += (
                        self.variables[f"xm_plus[{raiz_id},{s},{n}]"] == target_volume,
                        f"heuristica_compra_raiz_{s}_{n}",
                    )
                    self.lp_model += (
                        self.variables[f"xm_minus[{raiz_id},{s},{n}]"] == 0.0,
                        f"heuristica_sem_venda_raiz_{s}_{n}",
                    )

        # Trava produtos trimestrais e semestrais na raiz
        for s in self.SUBMERCADOS:
            for q in range(self.n_quarters):
                if f"xt_plus[{raiz_id},{s},{q}]" in self.variables:
                    self.lp_model += (self.variables[f"xt_plus[{raiz_id},{s},{q}]"] == 0.0, f"heur_xt_p_{s}_{q}")
                    self.lp_model += (self.variables[f"xt_minus[{raiz_id},{s},{q}]"] == 0.0, f"heur_xt_m_{s}_{q}")
            for h in range(self.n_semesters):
                if f"xs_plus[{raiz_id},{s},{h}]" in self.variables:
                    self.lp_model += (self.variables[f"xs_plus[{raiz_id},{s},{h}]"] == 0.0, f"heur_xs_p_{s}_{h}")
                    self.lp_model += (self.variables[f"xs_minus[{raiz_id},{s},{h}]"] == 0.0, f"heur_xs_m_{s}_{h}")

        # Trava nós filhos
        for nid, node in self.arvore.nodes.items():
            if nid == raiz_id:
                continue
            for s in self.SUBMERCADOS:
                for n in range(self.n_months):
                    if f"xm_plus[{nid},{s},{n}]" in self.variables:
                        self.lp_model += (self.variables[f"xm_plus[{nid},{s},{n}]"] == 0.0, f"heur_no_buy_{nid}_{s}_{n}")
                        self.lp_model += (self.variables[f"xm_minus[{nid},{s},{n}]"] == 0.0, f"heur_no_sell_{nid}_{s}_{n}")
                for q in range(self.n_quarters):
                    if f"xt_plus[{nid},{s},{q}]" in self.variables:
                        self.lp_model += (self.variables[f"xt_plus[{nid},{s},{q}]"] == 0.0, f"heur_no_xt_p_{nid}_{s}_{q}")
                        self.lp_model += (self.variables[f"xt_minus[{nid},{s},{q}]"] == 0.0, f"heur_no_xt_m_{nid}_{s}_{q}")
                for h in range(self.n_semesters):
                    if f"xs_plus[{nid},{s},{h}]" in self.variables:
                        self.lp_model += (self.variables[f"xs_plus[{nid},{s},{h}]"] == 0.0, f"heur_no_xs_p_{nid}_{s}_{h}")
                        self.lp_model += (self.variables[f"xs_minus[{nid},{s},{h}]"] == 0.0, f"heur_no_xs_m_{nid}_{s}_{h}")

    def definir_funcao_objetivo(self):
        """
        Constrói o fluxo financeiro real em R$ (multiplicando energia MWm pelas horas do mês):
        Receita Total = Receita Clientes (preço_venda * D * horas)
                      + Receita Vendas Contratos (P_v * Vendas * horas)
                      - Custo Compras Contratos (P_c * Compras * horas)
                      + Liquidação Spot (PLD * y_plus * horas)
                      - Liquidação Spot (PLD * y_minus * horas)
                      - Encargos Setoriais (encargo_mwh * D * horas)
                      [- Custo de Modulação Horária (gamma - 1) * PLD * D * horas, se informado]
        """
        self.expressao_fluxo_nos = {}

        for nid, node in self.arvore.nodes.items():
            t = node.mes_decisao
            expr_no = pulp.LpAffineExpression()

            for s in self.SUBMERCADOS:
                precos_t = self.precos_termo[nid][s]
                pld_no = self._obter_pld_no(node, s)
                demanda_no = self._obter_demanda_no(node, s)

                # 1. Contratos mensais (apenas existentes no nó)
                for n in range(self.n_months):
                    if f"xm_plus[{nid},{s},{n}]" in self.variables:
                        h_n = float(self.horas_mes[n])
                        pv = float(precos_t["pm_v"][n])
                        pc = float(precos_t["pm_c"][n])
                        expr_no += (pv * h_n) * self.variables[f"xm_minus[{nid},{s},{n}]"]
                        expr_no -= (pc * h_n) * self.variables[f"xm_plus[{nid},{s},{n}]"]

                # 2. Contratos trimestrais (apenas existentes no nó)
                for q in range(self.n_quarters):
                    if f"xt_plus[{nid},{s},{q}]" in self.variables:
                        start_m = q * 3
                        h_q = float(np.sum(self.horas_mes[start_m:start_m + 3]))
                        pv = float(precos_t["pt_v"][q])
                        pc = float(precos_t["pt_c"][q])
                        expr_no += (pv * h_q) * self.variables[f"xt_minus[{nid},{s},{q}]"]
                        expr_no -= (pc * h_q) * self.variables[f"xt_plus[{nid},{s},{q}]"]

                # 3. Contratos semestrais (apenas existentes no nó)
                for h in range(self.n_semesters):
                    if f"xs_plus[{nid},{s},{h}]" in self.variables:
                        start_m = h * 6
                        h_h = float(np.sum(self.horas_mes[start_m:start_m + 6]))
                        pv = float(precos_t["ps_v"][h])
                        pc = float(precos_t["ps_c"][h])
                        expr_no += (pv * h_h) * self.variables[f"xs_minus[{nid},{s},{h}]"]
                        expr_no -= (pc * h_h) * self.variables[f"xs_plus[{nid},{s},{h}]"]

                # 4. Liquidação física, receita de clientes e encargos nos meses governados pelo nó
                for m in node.meses:
                    h_m = float(self.horas_mes[m])
                    pld_m = float(pld_no[m])
                    d_m = float(demanda_no[m])

                    # Receita fixa de suprimento aos clientes livres
                    receita_clientes = float(self.preco_venda_cliente_mwh * d_m * h_m)
                    expr_no += receita_clientes

                    # Spot no submercado s
                    yp = self.variables[f"y_plus[{nid},{s},{m}]"]
                    ym = self.variables[f"y_minus[{nid},{s},{m}]"]
                    expr_no += (pld_m * h_m) * yp
                    expr_no -= (pld_m * h_m) * ym

                    # Custo/ganho de modulação horária intra-mensal frente a contrato flat
                    if self.fator_captura_demanda is not None:
                        gamma_m = float(self.fator_captura_demanda[s][m])
                        expr_no -= (gamma_m - 1.0) * pld_m * d_m * h_m

                    # Encargo regulatório ESS/EER cobrado sobre o consumo
                    custo_encargo = float(self.encargo_ccee_mwh * d_m * h_m)
                    expr_no -= custo_encargo

            # Fluxos opcionais de contratos cruzados NE->SE (precificados no NE) e FTRs
            if self.permitir_contrato_cruzado:
                precos_ne = self.precos_termo[nid]["NE"]
                pld_se_no = self._obter_pld_no(node, "SE")
                pld_ne_no = self._obter_pld_no(node, "NE")
                for n in range(self.n_months):
                    if f"xmc_plus[{nid},{n}]" in self.variables:
                        h_n = float(self.horas_mes[n])
                        expr_no += (float(precos_ne["pm_v"][n]) * h_n) * self.variables[f"xmc_minus[{nid},{n}]"]
                        expr_no -= (float(precos_ne["pm_c"][n]) * h_n) * self.variables[f"xmc_plus[{nid},{n}]"]
                    if self.custo_ftr_mwh is not None and f"ftr_plus[{nid},{n}]" in self.variables:
                        h_n = float(self.horas_mes[n])
                        p_ftr = float(self.precos_termo[nid]["_ftr_esp"][n]) + self.custo_ftr_mwh
                        expr_no -= (p_ftr * h_n) * self.variables[f"ftr_plus[{nid},{n}]"]
                for q in range(self.n_quarters):
                    if f"xtc_plus[{nid},{q}]" in self.variables:
                        start_m = q * 3
                        h_q = float(np.sum(self.horas_mes[start_m:start_m + 3]))
                        expr_no += (float(precos_ne["pt_v"][q]) * h_q) * self.variables[f"xtc_minus[{nid},{q}]"]
                        expr_no -= (float(precos_ne["pt_c"][q]) * h_q) * self.variables[f"xtc_plus[{nid},{q}]"]
                for h in range(self.n_semesters):
                    if f"xsc_plus[{nid},{h}]" in self.variables:
                        start_m = h * 6
                        h_h = float(np.sum(self.horas_mes[start_m:start_m + 6]))
                        expr_no += (float(precos_ne["ps_v"][h]) * h_h) * self.variables[f"xsc_minus[{nid},{h}]"]
                        expr_no -= (float(precos_ne["ps_c"][h]) * h_h) * self.variables[f"xsc_plus[{nid},{h}]"]
                if self.custo_ftr_mwh is not None:
                    for m in node.meses:
                        h_m = float(self.horas_mes[m])
                        spread_pos = max(0.0, float(pld_se_no[m]) - float(pld_ne_no[m]))
                        expr_no += (spread_pos * h_m) * self.variables[f"qftr[{nid},{m}]"]

            self.expressao_fluxo_nos[nid] = expr_no

        # Lucro acumulado terminal por folha R_ell (incluindo penalidade de lastro se ativa)
        self.expressao_lucro_folhas = {}
        termo_esperado = pulp.LpAffineExpression()
        raiz = self.arvore.root
        n_min_raiz = (max(raiz.meses) + 1) if raiz.meses else 1
        use_12m = self.lastro_janela_movel and self.n_months >= n_min_raiz + 12

        for leaf in self.arvore.leaves:
            caminho = self.arvore.caminho_ate_raiz(leaf.id)
            r_leaf = pulp.lpSum(self.expressao_fluxo_nos[anc] for anc in caminho)

            if self.penalidade_lastro_mwh > 0.0:
                mes_para_anc = {}
                for anc in caminho:
                    for m_anc in self.arvore.nodes[anc].meses:
                        mes_para_anc[m_anc] = anc

                m_audit_start = (n_min_raiz + 11) if use_12m else n_min_raiz
                for m_aud in range(m_audit_start, self.n_months):
                    d_var = self.deficit_lastro_vars[(leaf.id, m_aud)]
                    janela = range(m_aud - 11, m_aud + 1) if use_12m else [m_aud]
                    divisor = 12.0 if use_12m else 1.0
                    deficit_expr = pulp.LpAffineExpression()
                    for k_m in janela:
                        anc_k = mes_para_anc[k_m]
                        nd_k = self.arvore.nodes[anc_k]
                        for s in self.SUBMERCADOS:
                            d_k = float(self._obter_demanda_no(nd_k, s)[k_m])
                            e_k = float(self.E_base[s][k_m])
                            q_k = self.variables[f"q[{anc_k},{s},{k_m}]"]
                            qc_k = self.variables[f"qc[{anc_k},{k_m}]"] if (self.permitir_contrato_cruzado and s == "NE") else 0.0
                            deficit_expr += (1.0 / divisor) * (d_k - e_k - q_k - qc_k)
                    self.lp_model += (
                        d_var >= deficit_expr,
                        f"lastro_deficit_{leaf.id}_{m_aud}",
                    )
                    r_leaf -= (self.penalidade_lastro_mwh * float(self.horas_mes[m_aud])) * d_var

            self.expressao_lucro_folhas[leaf.id] = r_leaf

            termo_esperado += leaf.prob * r_leaf

            # Restrição CVaR: u_ell >= -R_leaf - alfa
            u_var = self.u_vars[leaf.id]
            self.lp_model += (
                u_var >= -r_leaf - self.alfa,
                f"cvar_u_{leaf.id}",
            )

        fator_beta = 1.0 / (1.0 - self.beta)
        termo_cvar = self.alfa + fator_beta * pulp.lpSum(
            leaf.prob * self.u_vars[leaf.id] for leaf in self.arvore.leaves
        )

        # FO = max (1 - lambda) * E[R] - lambda * CVaR_perda
        self.lp_model.setObjective(
            (1.0 - self._lambda) * termo_esperado - self._lambda * termo_cvar
        )

    def resolver(self, msg: bool = False, time_limit: Optional[int] = 60) -> int:
        """Resolve a instância LP usando HiGHS ou CBC."""
        solver = None
        if self.solver_name.upper() == "HIGHS" and hasattr(pulp, "HiGHS"):
            solver = pulp.HiGHS(msg=msg, timeLimit=time_limit)
        elif hasattr(pulp, "PULP_CBC_CMD"):
            solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)

        status = self.lp_model.solve(solver)

        if status == pulp.LpStatusOptimal:
            self.obj_value = float(pulp.value(self.lp_model.objective))

            # Lucro realizado em cada folha
            self.lucro_folhas = {
                leaf.id: float(pulp.value(self.expressao_lucro_folhas[leaf.id]))
                for leaf in self.arvore.leaves
            }

            # Retorno esperado
            self.expected_profit = float(
                sum(leaf.prob * self.lucro_folhas[leaf.id] for leaf in self.arvore.leaves)
            )

            # Cálculo canônico do CVaR das perdas a partir da distribuição obtida
            perdas = np.array([-self.lucro_folhas[leaf.id] for leaf in self.arvore.leaves], dtype=float)
            probs = np.array([leaf.prob for leaf in self.arvore.leaves], dtype=float)
            probs /= probs.sum()

            cvar_candidatos = [
                a + (1.0 / (1.0 - self.beta)) * np.sum(probs * np.maximum(0.0, perdas - a))
                for a in perdas
            ]
            self.cvar_perda = float(np.min(cvar_candidatos))

            idx_min = int(np.argmin(cvar_candidatos))
            self.var_perda_alfa = float(perdas[idx_min])

            # Captura de posições líquidas e fluxos
            for nid, node in self.arvore.nodes.items():
                t = node.mes_decisao
                self.fluxo_nos[nid] = float(pulp.value(self.expressao_fluxo_nos[nid]))
                for s in self.SUBMERCADOS:
                    for n in range(t, self.n_months):
                        var_q = self.variables.get(f"q[{nid},{s},{n}]")
                        if var_q is not None and var_q.varValue is not None:
                            self.posicao_nos[(nid, s, n)] = float(var_q.varValue)

        return status

    def calcular_metricas_regulatorias(self, vr_mwh: float = 231.47) -> Dict[str, float]:
        """
        Audita ex-post a conformidade de lastro (Decreto 5.163/2004) e exposição
        de curto prazo para garantias financeiras CCEE sobre a solução obtida.
        """
        raiz = self.arvore.root
        n_min_raiz = (max(raiz.meses) + 1) if raiz.meses else 1
        esp_def_mensal_gwh = 0.0
        esp_def_12m_gwh = 0.0
        prob_incump_12m = 0.0
        pico_curto_2m_vals = []
        probs = []

        for leaf in self.arvore.leaves:
            caminho = self.arvore.caminho_ate_raiz(leaf.id)
            mes_para_anc = {}
            for anc in caminho:
                for m_anc in self.arvore.nodes[anc].meses:
                    mes_para_anc[m_anc] = anc

            balanco_m = np.zeros(self.n_months, dtype=float)  # + = sobra, - = deficit MWm
            exp_spot_m = np.zeros(self.n_months, dtype=float)  # R$ exposição curta mensal
            for m_idx in range(self.n_months):
                anc_m = mes_para_anc[m_idx]
                nd_m = self.arvore.nodes[anc_m]
                h_m = float(self.horas_mes[m_idx])
                bal_mw = 0.0
                exp_r = 0.0
                for s in self.SUBMERCADOS:
                    d_m = float(self._obter_demanda_no(nd_m, s)[m_idx])
                    e_m = float(self.E_base[s][m_idx])
                    q_m = float(self.posicao_nos.get((anc_m, s, m_idx), 0.0))
                    qc_m = float(self.variables[f"qc[{anc_m},{m_idx}]"].varValue or 0.0) if (self.permitir_contrato_cruzado and s == "NE" and f"qc[{anc_m},{m_idx}]" in self.variables) else 0.0
                    dif_mw = (e_m + q_m + qc_m) - d_m
                    bal_mw += dif_mw
                    if dif_mw < 0.0:
                        pld_s_m = float(self._obter_pld_no(nd_m, s)[m_idx])
                        exp_r += (-dif_mw) * pld_s_m * h_m
                balanco_m[m_idx] = bal_mw
                exp_spot_m[m_idx] = exp_r

            # Déficit mensal em meses contratáveis (n >= n_min_raiz)
            def_mensal_mwh = sum(
                max(0.0, -balanco_m[m_idx]) * float(self.horas_mes[m_idx])
                for m_idx in range(n_min_raiz, self.n_months)
            )
            esp_def_mensal_gwh += leaf.prob * (def_mensal_mwh / 1e3)

            # Déficit em janela móvel de 12 meses nos meses totalmente contratáveis (m >= n_min_raiz + 11)
            def_12m_mwh = 0.0
            if self.n_months >= n_min_raiz + 12:
                for m_aud in range(n_min_raiz + 11, self.n_months):
                    avg_def_mw = max(0.0, float(-np.mean(balanco_m[m_aud - 11:m_aud + 1])))
                    def_12m_mwh += avg_def_mw * float(self.horas_mes[m_aud])
            else:
                def_12m_mwh = def_mensal_mwh

            esp_def_12m_gwh += leaf.prob * (def_12m_mwh / 1e3)
            if def_12m_mwh > 1e-3:
                prob_incump_12m += leaf.prob

            # Pico de exposição spot curta em 2 meses consecutivos (proxy de chamada de garantias CCEE)
            if self.n_months >= 2:
                exp_2m = [exp_spot_m[k] + exp_spot_m[k + 1] for k in range(self.n_months - 1)]
                pico_curto_2m_vals.append(float(np.max(exp_2m)) / 1e6)
            else:
                pico_curto_2m_vals.append(float(exp_spot_m[0]) / 1e6)
            probs.append(leaf.prob)

        probs_arr = np.asarray(probs, dtype=float)
        probs_arr /= probs_arr.sum()
        pico_arr = np.asarray(pico_curto_2m_vals, dtype=float)
        idx_sort = np.argsort(pico_arr)
        cdf = np.cumsum(probs_arr[idx_sort])
        pico_p95 = float(pico_arr[idx_sort][np.searchsorted(cdf, 0.95, side="left")])

        return {
            "deficit_mensal_esp_gwh": float(esp_def_mensal_gwh),
            "deficit_12m_esp_gwh": float(esp_def_12m_gwh),
            "penalidade_vr_implicita_M": float(esp_def_12m_gwh * vr_mwh / 1e3),
            "prob_incumprimento_12m_pct": float(prob_incump_12m * 100.0),
            "garantia_pico_2m_esp_M": float(np.sum(probs_arr * pico_arr)),
            "garantia_pico_2m_p95_M": pico_p95,
        }

