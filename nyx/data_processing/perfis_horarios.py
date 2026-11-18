"""
Perfis horários para a extensão de modulação horária do NYX.

Três insumos, todos em base de 24 horas por mês do calendário (1..12):

1. Perfis típicos de PLD (fi[c, k, h]): K-means sobre o formato diário do PLD
   horário da CCEE (PLD da hora / média do dia), por mês do calendário. Média
   em h de cada perfil = 1, de modo que o PLD médio mensal das séries não muda.
2. Perfis de geração (wG[c, h]): geração horária média do ONS por fonte
   (hidráulica, eólica, solar), normalizada para somar 1 no dia.
3. Perfis de consumo (wD[c, h]): carga horária do ONS por subsistema ou perfis
   sintéticos por classe (industrial, comercial, residencial).

Cada cenário horário é o par (série s, amostra j). Em cada mês do horizonte a
amostra j sorteia um perfil típico de PLD segundo a frequência do cluster, e o
preço efetivo da liquidação vira PLD[m, s] x fator[m, s, j], com

    fator_V[m, s, j] = soma_h wG[c(m), h] x fi[c(m), k(m, s, j), h]
    fator_C[m, s, j] = soma_h wD[c(m), h] x fi[c(m), k(m, s, j), h]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ONS_DIR = DATA_DIR / "external" / "ons"
HORAS = 24

SUBMERCADO_CCEE = {"SE": "SUDESTE", "S": "SUL", "NE": "NORDESTE", "N": "NORTE"}

COLUNA_ONS = {
    "hidraulica": "val_gerhidraulica",
    "eolica": "val_gereolica",
    "solar": "val_gersolar",
    "termica": "val_gertermica",
    "carga": "val_carga",
}

# Perfis sintéticos de consumo (valores relativos por hora, 0h..23h).
# Formatos estilizados; a CCEE não permitiu download automatizado da medição por classe.
PERFIS_CONSUMO_SINTETICOS = {
    "plano": [1.0] * HORAS,
    "industrial": [
        0.55, 0.52, 0.50, 0.50, 0.52, 0.60,
        0.85, 1.15, 1.30, 1.35, 1.35, 1.30,
        1.20, 1.30, 1.35, 1.35, 1.30, 1.15,
        0.95, 0.85, 0.80, 0.72, 0.65, 0.60,
    ],
    "comercial": [
        0.45, 0.42, 0.40, 0.40, 0.42, 0.48,
        0.60, 0.85, 1.20, 1.40, 1.50, 1.55,
        1.50, 1.55, 1.65, 1.65, 1.55, 1.35,
        1.05, 0.85, 0.72, 0.62, 0.55, 0.50,
    ],
    "residencial": [
        0.62, 0.55, 0.52, 0.50, 0.52, 0.62,
        0.85, 0.95, 0.85, 0.78, 0.78, 0.82,
        0.88, 0.85, 0.82, 0.85, 0.95, 1.20,
        1.60, 1.75, 1.70, 1.50, 1.15, 0.85,
    ],
}


@dataclass
class PerfisHorarios:
    """Perfis horários por mês do calendário e sorteio de perfis de PLD."""

    fi: dict  # c -> array (K_c, 24), média em h = 1
    prob: dict  # c -> array (K_c,), soma = 1
    w_geracao: np.ndarray  # (12, 24), soma em h = 1
    w_consumo: np.ndarray  # (12, 24), soma em h = 1
    nome_geracao: str = "plano"
    nome_consumo: str = "plano"
    descricao: dict = field(default_factory=dict)

    def fatores(
        self,
        n_meses,
        n_series,
        n_amostras,
        mes_calendario_m0,
        seed=2026,
        pld=None,
        pld_min=None,
        pld_max_horario=None,
    ):
        """Retorna (fator_V, fator_C, k_sorteado), cada um com forma (n_meses, n_series, n_amostras).

        Se `pld` (n_meses, n_series) e os limites por mês (`pld_min`,
        `pld_max_horario`, vetores de n_meses) forem dados, o preço de cada hora
        PLD[m, s] x fi[h] é limitado a [PLD mínimo, PLD máximo horário] antes de
        ponderar pelo perfil, e o fator passa a depender também do nível do PLD:
            fator[m, s, j] = soma_h w[h] x clip(PLD[m, s] x fi[k, h]) / PLD[m, s]
        Sem esses argumentos o comportamento é o original (sem limites horários).
        """
        rng = np.random.default_rng(seed)
        fator_v = np.ones((n_meses, n_series, n_amostras))
        fator_c = np.ones((n_meses, n_series, n_amostras))
        k_sorteado = np.zeros((n_meses, n_series, n_amostras), dtype=int)
        limitar = pld is not None and pld_min is not None and pld_max_horario is not None
        for m in range(n_meses):
            c = calendario(mes_calendario_m0, m)
            fi_c = self.fi[c]
            ks = rng.choice(len(self.prob[c]), size=(n_series, n_amostras), p=self.prob[c])
            k_sorteado[m] = ks
            if not limitar:
                fv_k = fi_c @ self.w_geracao[c - 1]
                fc_k = fi_c @ self.w_consumo[c - 1]
                fator_v[m] = fv_k[ks]
                fator_c[m] = fc_k[ks]
                continue
            nivel = np.asarray(pld[m], dtype=float)[:n_series]  # (S,)
            base = np.where(nivel > 0, nivel, 1.0)
            preco_h = np.clip(
                base[:, None, None] * fi_c[None, :, :],
                float(pld_min[m]),
                float(pld_max_horario[m]),
            )  # (S, K, 24)
            fv_sk = (preco_h @ self.w_geracao[c - 1]) / base[:, None]
            fc_sk = (preco_h @ self.w_consumo[c - 1]) / base[:, None]
            fv_sk[nivel <= 0] = 1.0
            fc_sk[nivel <= 0] = 1.0
            fator_v[m] = np.take_along_axis(fv_sk, ks, axis=1)
            fator_c[m] = np.take_along_axis(fc_sk, ks, axis=1)
        return fator_v, fator_c, k_sorteado


def calendario(mes_calendario_m0, m):
    """Mês do calendário (1..12) do mês m do horizonte."""
    return (mes_calendario_m0 - 1 + m) % 12 + 1


def _normalizar_soma(v):
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    total = v.sum()
    if total <= 0:
        return np.full(HORAS, 1.0 / HORAS)
    return v / total


# ---------------------------------------------------------------------------
# PLD horário (CCEE)
# ---------------------------------------------------------------------------

def carregar_pld_horario(anos, submercado="SE"):
    """PLD horário da CCEE: colunas data, mes, hora, pld."""
    nome = SUBMERCADO_CCEE.get(submercado, submercado)
    frames = []
    for ano in anos:
        path = DATA_DIR / f"pld_horario_{ano}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str)
        df = df[df["SUBMERCADO"].str.strip().str.upper() == nome]
        mes_ref = df["MES_REFERENCIA"].str.strip()
        out = pd.DataFrame(
            {
                "ano": mes_ref.str[:4].astype(int),
                "mes": mes_ref.str[4:6].astype(int),
                "dia": df["DIA"].astype(int),
                "hora": df["HORA"].astype(int),
                "pld": pd.to_numeric(df["PLD_HORA"].str.replace(",", "."), errors="coerce"),
            }
        )
        frames.append(out)
    if not frames:
        raise FileNotFoundError(f"Nenhum arquivo pld_horario_<ano>.csv encontrado para {anos}.")
    pld = pd.concat(frames, ignore_index=True).dropna(subset=["pld"])
    pld["data"] = pd.to_datetime(dict(year=pld.ano, month=pld.mes, day=pld.dia))
    return pld.sort_values(["data", "hora"]).reset_index(drop=True)


def matriz_diaria(df, coluna):
    """Pivot dia x hora (24 colunas), descartando dias incompletos."""
    piv = df.pivot_table(index="data", columns="hora", values=coluna, aggfunc="mean")
    piv = piv.reindex(columns=range(HORAS))
    return piv.dropna()


def kmeans(x, k, seed=7, n_init=10, max_iter=300):
    """K-means (Lloyd) com inicialização k-means++; retorna (centroides, rótulos, inércia)."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = min(k, n)
    melhor = None
    for _ in range(n_init):
        centros = [x[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.array(centros)[None]) ** 2).sum(-1), axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            centros.append(x[rng.choice(n, p=p)])
        centros = np.array(centros)
        for _ in range(max_iter):
            dist = ((x[:, None, :] - centros[None]) ** 2).sum(-1)
            rotulos = dist.argmin(1)
            novos = np.array(
                [x[rotulos == j].mean(0) if np.any(rotulos == j) else centros[j] for j in range(k)]
            )
            if np.allclose(novos, centros):
                break
            centros = novos
        inercia = float(((x - centros[rotulos]) ** 2).sum())
        if melhor is None or inercia < melhor[2]:
            melhor = (centros, rotulos, inercia)
    return melhor


def perfis_pld_kmeans(anos, n_perfis, submercado="SE", seed=7):
    """Perfis típicos de PLD por mês do calendário.

    Retorna (fi, prob, formatos_diarios): fi[c] com forma (K, 24) e média em h = 1;
    prob[c] é a frequência de cada cluster; formatos_diarios[c] guarda os dias usados.
    """
    pld = carregar_pld_horario(anos, submercado)
    fi, prob, formatos = {}, {}, {}
    for c in range(1, 13):
        piv = matriz_diaria(pld[pld.mes == c], "pld")
        media_dia = piv.mean(axis=1)
        piv = piv[media_dia > 1.0]
        forma = piv.div(piv.mean(axis=1), axis=0).to_numpy()
        centros, rotulos, _ = kmeans(forma, n_perfis, seed=seed + c)
        centros = centros / centros.mean(axis=1, keepdims=True)
        freq = np.bincount(rotulos, minlength=len(centros)).astype(float)
        ordem = np.argsort(-freq)
        fi[c] = centros[ordem]
        prob[c] = freq[ordem] / freq.sum()
        formatos[c] = forma
    return fi, prob, formatos


def perfil_pld_plano():
    fi = {c: np.ones((1, HORAS)) for c in range(1, 13)}
    prob = {c: np.ones(1) for c in range(1, 13)}
    return fi, prob


# ---------------------------------------------------------------------------
# Geração e carga (ONS)
# ---------------------------------------------------------------------------

def carregar_balanco_ons(anos):
    frames = []
    for ano in anos:
        path = ONS_DIR / f"BALANCO_ENERGIA_SUBSISTEMA_{ano}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, sep=";"))
    if not frames:
        raise FileNotFoundError(f"Balanço horário do ONS não encontrado em {ONS_DIR}.")
    df = pd.concat(frames, ignore_index=True)
    df["instante"] = pd.to_datetime(df["din_instante"])
    df["data"] = df["instante"].dt.normalize()
    df["mes"] = df["instante"].dt.month
    df["hora"] = df["instante"].dt.hour
    return df


def perfil_ons(fonte, subsistema, anos):
    """Perfil médio (12, 24) de uma fonte do ONS, normalizado para somar 1 em cada mês."""
    df = carregar_balanco_ons(anos)
    df = df[df["id_subsistema"] == subsistema]
    col = COLUNA_ONS[fonte]
    media = df.groupby(["mes", "hora"])[col].mean().unstack("hora").reindex(range(1, 13))
    return np.vstack([_normalizar_soma(media.loc[c].to_numpy()) for c in range(1, 13)])


def perfil_csv(caminho):
    """Perfil de um CSV com colunas hora (0..23), valor e, opcionalmente, mes (1..12)."""
    df = pd.read_csv(caminho)
    if "mes" not in df.columns:
        base = df.sort_values("hora")["valor"].to_numpy()
        return np.vstack([_normalizar_soma(base)] * 12)
    piv = df.pivot_table(index="mes", columns="hora", values="valor").reindex(range(1, 13))
    return np.vstack([_normalizar_soma(piv.loc[c].to_numpy()) for c in range(1, 13)])


def resolver_perfil(nome, anos_ons):
    """Resolve o nome do perfil em uma matriz (12, 24).

    Nomes aceitos: plano | hidraulica | eolica | solar | carga (ONS, subsistema padrão)
    | <fonte>_<subsistema> (ex.: solar_NE, carga_SE) | industrial | comercial
    | residencial | csv:<arquivo em data/>
    """
    if nome.startswith("csv:"):
        return perfil_csv(DATA_DIR / nome[4:])
    if nome in PERFIS_CONSUMO_SINTETICOS:
        return np.vstack([_normalizar_soma(PERFIS_CONSUMO_SINTETICOS[nome])] * 12)
    subsistema_padrao = {"hidraulica": "SE", "eolica": "NE", "solar": "NE", "termica": "SE", "carga": "SE"}
    fonte, _, sub = nome.partition("_")
    if fonte not in COLUNA_ONS:
        raise ValueError(f"Perfil desconhecido: {nome}")
    try:
        return perfil_ons(fonte, sub.upper() or subsistema_padrao[fonte], anos_ons)
    except FileNotFoundError:
        return np.vstack([_normalizar_soma([1.0] * HORAS)] * 12)


def montar_perfis(
    perfil_geracao="plano",
    perfil_consumo="plano",
    n_perfis_pld=6,
    anos_pld=(2024, 2025),
    anos_ons=(2024, 2025),
    submercado="SE",
    pld_plano=False,
):
    if pld_plano:
        fi, prob = perfil_pld_plano()
    else:
        fi, prob, _ = perfis_pld_kmeans(anos_pld, n_perfis_pld, submercado)
    return PerfisHorarios(
        fi=fi,
        prob=prob,
        w_geracao=resolver_perfil(perfil_geracao, anos_ons),
        w_consumo=resolver_perfil(perfil_consumo, anos_ons),
        nome_geracao=perfil_geracao,
        nome_consumo=perfil_consumo,
        descricao={
            "n_perfis_pld": n_perfis_pld,
            "anos_pld": list(anos_pld),
            "anos_ons": list(anos_ons),
            "submercado_pld": submercado,
            "pld_plano": pld_plano,
        },
    )


def fator_captura_historico(fonte, subsistema, anos, submercado_pld="SE"):
    """Fator de captura realizado por mês: soma(g_h x PLD_h) / (soma(g_h) x PLD médio do mês)."""
    pld = carregar_pld_horario(anos, submercado_pld)
    ons = carregar_balanco_ons(anos)
    ons = ons[ons["id_subsistema"] == subsistema][["data", "hora", COLUNA_ONS[fonte]]]
    df = pld.merge(ons, on=["data", "hora"], how="inner")
    df["periodo"] = df["data"].dt.to_period("M")
    col = COLUNA_ONS[fonte]

    def _fator(g):
        pld_medio = g["pld"].mean()
        energia = g[col].sum()
        if pld_medio <= 0 or energia <= 0:
            return np.nan
        return (g[col] * g["pld"]).sum() / (energia * pld_medio)

    return df.groupby("periodo").apply(_fator).rename("fator_captura")
