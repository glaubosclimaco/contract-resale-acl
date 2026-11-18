"""
Geração e manipulação de árvores de cenários para o modelo multiestágio do NYX.

Estrutura da árvore:
- Cada nó nu possui:
  - id único (raiz tem id = 0);
  - pai a(nu) e filhos;
  - estágio t(nu);
  - meses governados pelo estágio (ex.: mês 0, depois meses 1..3, trimestres seguintes);
  - probabilidade incondicional pi_nu (com soma 1 por estágio);
  - probabilidade condicional dada o pai;
  - cenários da amostra original associados ao nó;
  - realizações esperadas (ou representativas) de PLD, excedente físico E
    e matrizes de preços de mercado (mensal, trimestral, semestral).

Algoritmos suportados:
- 'kmeans': agrupamento condicional por estágio via K-means++ sobre trajetórias;
- 'quantis': particionamento por quantis da métrica principal (PLD médio do estágio);
- 'forward_selection': seleção progressiva de cenários representativos (Heitsch–Römisch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class NoCenario:
    id: int
    parent_id: Optional[int]
    stage: int
    meses: List[int]
    prob: float
    cond_prob: float
    cenarios_indices: List[int]
    children_ids: List[int] = field(default_factory=list)
    pld: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    pld_se: Optional[np.ndarray] = None
    pld_ne: Optional[np.ndarray] = None
    demanda_fator: Optional[np.ndarray] = None
    e: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    precos_venda_m: Optional[np.ndarray] = None
    precos_compra_m: Optional[np.ndarray] = None
    precos_venda_t: Optional[np.ndarray] = None
    precos_compra_t: Optional[np.ndarray] = None
    precos_venda_s: Optional[np.ndarray] = None
    precos_compra_s: Optional[np.ndarray] = None
    fator_v: Optional[np.ndarray] = None
    fator_c: Optional[np.ndarray] = None

    @property
    def mes_decisao(self) -> int:
        """Primeiro mês do estágio, no qual as decisões são tomadas."""
        return self.meses[0] if self.meses else 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None


@dataclass
class ArvoreCenarios:
    nodes: Dict[int, NoCenario]
    stages: List[List[int]]
    n_months: int
    n_series_originais: int
    divisao_estagios: List[List[int]]

    @property
    def root(self) -> NoCenario:
        return self.nodes[0]

    @property
    def leaves(self) -> List[NoCenario]:
        last_stage = self.stages[-1]
        return [self.nodes[nid] for nid in last_stage]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    def caminho_ate_raiz(self, node_id: int) -> List[int]:
        """Retorna lista de IDs de nós do caminho da raiz até node_id (inclusivo)."""
        caminho = []
        cur: Optional[int] = node_id
        while cur is not None:
            caminho.append(cur)
            cur = self.nodes[cur].parent_id
        caminho.reverse()
        return caminho

    def nos_do_estagio(self, stage: int) -> List[NoCenario]:
        return [self.nodes[nid] for nid in self.stages[stage]]

    def verificar_consistencia(self, tol: float = 1e-6) -> Dict[str, Union[bool, float, str]]:
        """Valida que a árvore é matematicamente consistente."""
        # 1. Soma das probabilidades por estágio deve ser 1.0
        erros_soma = {}
        for s_idx, nids in enumerate(self.stages):
            soma = sum(self.nodes[nid].prob for nid in nids)
            if abs(soma - 1.0) > tol:
                erros_soma[s_idx] = soma

        # 2. Probabilidades condicionais somam 1 para os filhos de cada nó não folha
        erros_filhos = {}
        for nid, node in self.nodes.items():
            if node.children_ids:
                soma_filhos = sum(self.nodes[cid].cond_prob for cid in node.children_ids)
                if abs(soma_filhos - 1.0) > tol:
                    erros_filhos[nid] = soma_filhos

        # 3. Propagação de probabilidade incondicional: prob(filho) = prob(pai) * cond_prob(filho)
        erros_incond = {}
        for nid, node in self.nodes.items():
            if node.parent_id is not None:
                pai = self.nodes[node.parent_id]
                esperado = pai.prob * node.cond_prob
                if abs(node.prob - esperado) > tol:
                    erros_incond[nid] = (node.prob, esperado)

        consistente = (
            len(erros_soma) == 0 and len(erros_filhos) == 0 and len(erros_incond) == 0
        )
        return {
            "consistente": consistente,
            "erros_soma_estagios": erros_soma,
            "erros_soma_filhos": erros_filhos,
            "erros_propagacao": erros_incond,
            "n_nodes": self.n_nodes,
            "n_stages": self.n_stages,
            "n_leaves": self.n_leaves,
        }

    def calcular_estatisticas_pld(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula média ponderada, percentil 5 e percentil 95 do PLD por mês nas folhas."""
        leaves = self.leaves
        n_folhas = len(leaves)
        pld_mat = np.zeros((self.n_months, n_folhas), dtype=float)
        probs = np.zeros(n_folhas, dtype=float)

        for j, leaf in enumerate(leaves):
            probs[j] = leaf.prob
            pld_mat[:, j] = leaf.pld

        # Normalizar probabilidades caso haja pequeno desvio numérico
        probs = probs / probs.sum()

        media = pld_mat @ probs
        p5 = np.zeros(self.n_months, dtype=float)
        p95 = np.zeros(self.n_months, dtype=float)

        # Cálculo de quantis ponderados
        for m in range(self.n_months):
            ordenado_idx = np.argsort(pld_mat[m, :])
            valores = pld_mat[m, ordenado_idx]
            cum_probs = np.cumsum(probs[ordenado_idx])
            p5[m] = valores[np.searchsorted(cum_probs, 0.05)]
            p95[m] = valores[min(np.searchsorted(cum_probs, 0.95), n_folhas - 1)]

        return media, p5, p95

    def resumo(self) -> str:
        linhas = [
            f"Árvore de Cenários: {self.n_stages} estágios, {self.n_nodes} nós, {self.n_leaves} folhas.",
            f"Séries originais cobertas: {self.n_series_originais}",
        ]
        for s, nids in enumerate(self.stages):
            meses = self.divisao_estagios[s]
            linhas.append(
                f"  Estágio {s:2d} (meses {meses}): {len(nids):4d} nós, "
                f"prob total = {sum(self.nodes[nid].prob for nid in nids):.4f}"
            )
        return "\n".join(linhas)


def _kmeans_1d_ou_nd(
    x: np.ndarray, k: int, seed: int = 7, n_init: int = 10, max_iter: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means determinístico com K-means++ em NumPy puro."""
    n = x.shape[0]
    k = min(k, n)
    if k <= 1 or n <= 1:
        centros = x.mean(axis=0, keepdims=True)
        rotulos = np.zeros(n, dtype=int)
        return centros, rotulos

    rng = np.random.default_rng(seed)
    melhor_inercia = float("inf")
    melhores_centros = None
    melhores_rotulos = None

    for _ in range(n_init):
        # Inicialização k-means++
        centros_list = [x[rng.integers(n)]]
        for _ in range(1, k):
            centros_arr = np.array(centros_list)
            dist_sq = ((x[:, None, :] - centros_arr[None, :, :]) ** 2).sum(axis=-1)
            min_dist_sq = np.min(dist_sq, axis=1)
            soma_d2 = min_dist_sq.sum()
            probs = min_dist_sq / soma_d2 if soma_d2 > 0 else np.full(n, 1.0 / n)
            proximo_idx = rng.choice(n, p=probs)
            centros_list.append(x[proximo_idx])

        centros = np.array(centros_list)

        # Iteração de Lloyd
        for _ in range(max_iter):
            dist_sq = ((x[:, None, :] - centros[None, :, :]) ** 2).sum(axis=-1)
            rotulos = dist_sq.argmin(axis=1)
            novos_centros = np.zeros_like(centros)
            for j in range(k):
                mask = rotulos == j
                if np.any(mask):
                    novos_centros[j] = x[mask].mean(axis=0)
                else:
                    novos_centros[j] = centros[j]

            if np.allclose(novos_centros, centros, atol=1e-7):
                break
            centros = novos_centros

        inercia = float(((x - centros[rotulos]) ** 2).sum())
        if inercia < melhor_inercia:
            melhor_inercia = inercia
            melhores_centros = centros
            melhores_rotulos = rotulos

    return melhores_centros, melhores_rotulos


def _particionar_cenarios(
    indices: List[int],
    features: np.ndarray,
    ramificacao: int,
    metodo: str = "kmeans",
    seed: int = 7,
) -> List[List[int]]:
    """Particiona uma lista de cenários em até ramificacao grupos."""
    n = len(indices)
    k = min(ramificacao, n)
    if k <= 1:
        return [indices]

    if metodo == "kmeans":
        sub_feat = features[indices]
        # Normalizar features para evitar dominância de escala
        std = sub_feat.std(axis=0)
        std[std == 0] = 1.0
        sub_feat_norm = (sub_feat - sub_feat.mean(axis=0)) / std

        _, rotulos = _kmeans_1d_ou_nd(sub_feat_norm, k=k, seed=seed)
        grupos: Dict[int, List[int]] = {j: [] for j in range(k)}
        for local_idx, r in enumerate(rotulos):
            grupos[r].append(indices[local_idx])
        # Filtrar grupos vazios
        resultado = [g for g in grupos.values() if len(g) > 0]
        return resultado if resultado else [indices]

    elif metodo == "quantis":
        # Ordena pelo primeiro componente da feature (ex.: PLD médio)
        valores = features[indices, 0]
        ordem = np.argsort(valores)
        ordenados = [indices[i] for i in ordem]
        # Divide em k blocos contíguos
        tamanhos = np.array_split(ordenados, k)
        return [list(t) for t in tamanhos if len(t) > 0]

    elif metodo == "forward_selection":
        # Heitsch-Römisch: seleciona k pontos representativos e associa os demais ao mais próximo
        sub_feat = features[indices]
        std = sub_feat.std(axis=0)
        std[std == 0] = 1.0
        norm_feat = (sub_feat - sub_feat.mean(axis=0)) / std

        # Ponto inicial: o mais próximo da média
        media = norm_feat.mean(axis=0)
        dist_media = ((norm_feat - media) ** 2).sum(axis=1)
        selecionados = [int(np.argmin(dist_media))]

        for _ in range(1, k):
            sel_arr = norm_feat[selecionados]
            dists = ((norm_feat[:, None, :] - sel_arr[None, :, :]) ** 2).sum(axis=-1)
            min_dists = np.min(dists, axis=1)
            # Escolhe o cenário com maior distância mínima para os já selecionados
            candidato = int(np.argmax(min_dists))
            if candidato not in selecionados:
                selecionados.append(candidato)

        # Associação por proximidade
        sel_arr = norm_feat[selecionados]
        dists = ((norm_feat[:, None, :] - sel_arr[None, :, :]) ** 2).sum(axis=-1)
        rotulos = dists.argmin(axis=1)

        grupos_fs: Dict[int, List[int]] = {j: [] for j in range(len(selecionados))}
        for local_idx, r in enumerate(rotulos):
            grupos_fs[r].append(indices[local_idx])
        return [g for g in grupos_fs.values() if len(g) > 0]

    else:
        raise ValueError(f"Método de particionamento desconhecido: {metodo}")


def gerar_divisao_estagios_padrao(n_months: int = 24) -> List[List[int]]:
    """Gera divisão em estágios: meses 0, 1, 2, 3 mensais e trimestres subsequentes."""
    if n_months <= 4:
        return [[m] for m in range(n_months)]

    estagios: List[List[int]] = []
    # Meses iniciais granulares (mensais)
    for m in range(min(4, n_months)):
        estagios.append([m])

    # Meses subsequentes agrupados em blocos de até 3 meses (trimestrais)
    cur = 4
    while cur < n_months:
        fim = min(cur + 3, n_months)
        estagios.append(list(range(cur, fim)))
        cur = fim

    return estagios


def construir_arvore_cenarios(
    PLD_series: np.ndarray,
    E_series: np.ndarray,
    estagios: Optional[List[List[int]]] = None,
    ramificacao: Union[int, List[int]] = 3,
    metodo: str = "kmeans",
    seed: int = 2026,
    MARKET_PRICES_series: Optional[np.ndarray] = None,
    MARKET_PRICES_BUY_series: Optional[np.ndarray] = None,
    PRECOS_T_series: Optional[np.ndarray] = None,
    PRECOS_T_BUY_series: Optional[np.ndarray] = None,
    PRECOS_S_series: Optional[np.ndarray] = None,
    PRECOS_S_BUY_series: Optional[np.ndarray] = None,
    fator_v_series: Optional[np.ndarray] = None,
    fator_c_series: Optional[np.ndarray] = None,
) -> ArvoreCenarios:
    """Constrói a árvore de cenários a partir das trajetórias amostradas.

    Parâmetros:
    -----------
    PLD_series: array (n_months, n_series)
    E_series: array (n_months, n_series)
    estagios: lista de listas de meses de cada estágio (ex.: [[0], [1], [2], [3], [4,5,6], ...])
    ramificacao: int ou lista de int (fator de ramificação em cada estágio)
    metodo: 'kmeans', 'quantis' ou 'forward_selection'
    seed: semente pseudoaleatória
    """
    n_months, n_series = PLD_series.shape
    assert E_series.shape == (n_months, n_series), "Dimensões de E e PLD não coincidem"

    if estagios is None:
        divisao = gerar_divisao_estagios_padrao(n_months)
    else:
        divisao = estagios

    n_stages = len(divisao)

    # Vetor de ramificação por estágio
    if isinstance(ramificacao, int):
        ramif_lista = [ramificacao] * n_stages
    else:
        ramif_lista = list(ramificacao)
        if len(ramif_lista) < n_stages:
            ramif_lista += [ramif_lista[-1]] * (n_stages - len(ramif_lista))

    # Matriz de features por mês para cada cenário (PLD e E)
    # Shape: (n_series, n_months * 2)
    features_completas = np.column_stack([PLD_series.T, E_series.T])

    nodes: Dict[int, NoCenario] = {}
    stages_nodes: List[List[int]] = [[] for _ in range(n_stages)]

    def _computar_medias(indices: List[int]) -> Dict[str, np.ndarray]:
        sub_pld = PLD_series[:, indices].mean(axis=1)
        sub_e = E_series[:, indices].mean(axis=1)
        res = {"pld": sub_pld, "e": sub_e}

        if MARKET_PRICES_series is not None:
            res["precos_venda_m"] = MARKET_PRICES_series[:, :, indices].mean(axis=2)
        if MARKET_PRICES_BUY_series is not None:
            res["precos_compra_m"] = MARKET_PRICES_BUY_series[:, :, indices].mean(axis=2)
        if PRECOS_T_series is not None:
            res["precos_venda_t"] = PRECOS_T_series[:, :, indices].mean(axis=2)
        if PRECOS_T_BUY_series is not None:
            res["precos_compra_t"] = PRECOS_T_BUY_series[:, :, indices].mean(axis=2)
        if PRECOS_S_series is not None:
            res["precos_venda_s"] = PRECOS_S_series[:, :, indices].mean(axis=2)
        if PRECOS_S_BUY_series is not None:
            res["precos_compra_s"] = PRECOS_S_BUY_series[:, :, indices].mean(axis=2)
        if fator_v_series is not None:
            res["fator_v"] = fator_v_series[:, indices].mean(axis=1)
        if fator_c_series is not None:
            res["fator_c"] = fator_c_series[:, indices].mean(axis=1)
        return res

    # 1. Criação do nó raiz (estágio 0)
    todos_indices = list(range(n_series))
    dados_raiz = _computar_medias(todos_indices)
    raiz = NoCenario(
        id=0,
        parent_id=None,
        stage=0,
        meses=divisao[0],
        prob=1.0,
        cond_prob=1.0,
        cenarios_indices=todos_indices,
        pld=dados_raiz["pld"],
        e=dados_raiz["e"],
        precos_venda_m=dados_raiz.get("precos_venda_m"),
        precos_compra_m=dados_raiz.get("precos_compra_m"),
        precos_venda_t=dados_raiz.get("precos_venda_t"),
        precos_compra_t=dados_raiz.get("precos_compra_t"),
        precos_venda_s=dados_raiz.get("precos_venda_s"),
        precos_compra_s=dados_raiz.get("precos_compra_s"),
        fator_v=dados_raiz.get("fator_v"),
        fator_c=dados_raiz.get("fator_c"),
    )
    nodes[0] = raiz
    stages_nodes[0].append(0)

    proximo_id = 1

    # 2. Ramificação estágio por estágio
    for s in range(1, n_stages):
        meses_estagio = divisao[s]
        # Extrair features pertinentes a este estágio (PLD e E dos meses do estágio)
        colunas_estagio = list(meses_estagio) + [m + n_months for m in meses_estagio]
        features_estagio = features_completas[:, colunas_estagio]

        ramif = ramif_lista[s]
        pais_estagio = stages_nodes[s - 1]

        for pai_id in pais_estagio:
            pai = nodes[pai_id]
            sub_indices = pai.cenarios_indices

            particoes = _particionar_cenarios(
                indices=sub_indices,
                features=features_estagio,
                ramificacao=ramif,
                metodo=metodo,
                seed=seed + s * 1000 + pai_id,
            )

            for part in particoes:
                cond_p = len(part) / len(sub_indices)
                incond_p = pai.prob * cond_p
                dados_filho = _computar_medias(part)

                filho = NoCenario(
                    id=proximo_id,
                    parent_id=pai_id,
                    stage=s,
                    meses=meses_estagio,
                    prob=incond_p,
                    cond_prob=cond_p,
                    cenarios_indices=part,
                    pld=dados_filho["pld"],
                    e=dados_filho["e"],
                    precos_venda_m=dados_filho.get("precos_venda_m"),
                    precos_compra_m=dados_filho.get("precos_compra_m"),
                    precos_venda_t=dados_filho.get("precos_venda_t"),
                    precos_compra_t=dados_filho.get("precos_compra_t"),
                    precos_venda_s=dados_filho.get("precos_venda_s"),
                    precos_compra_s=dados_filho.get("precos_compra_s"),
                    fator_v=dados_filho.get("fator_v"),
                    fator_c=dados_filho.get("fator_c"),
                )
                nodes[proximo_id] = filho
                pai.children_ids.append(proximo_id)
                stages_nodes[s].append(proximo_id)
                proximo_id += 1

    arvore = ArvoreCenarios(
        nodes=nodes,
        stages=stages_nodes,
        n_months=n_months,
        n_series_originais=n_series,
        divisao_estagios=divisao,
    )

    return arvore
