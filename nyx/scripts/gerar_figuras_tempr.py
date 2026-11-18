"""
Gera figuras de alta resolução (300 DPI) para o artigo completo TEMPR:
1. fig_tempr_pld_spread.png: Evolução histórica do PLD SE vs NE e do spread regional (2021-2025).
2. fig_tempr_fronteira.png: Fronteira eficiente Mean-CVaR (Retorno Esperado vs CVaR).
3. fig_tempr_estabilidade.png: Estabilidade In-Sample (convergência do CV e valor da FO com n_folhas).
4. fig_tempr_backtest.png: Decomposição financeira no teste out-of-sample (2024-2025).
5. fig_tempr_sensibilidade.png: Sensibilidade aos encargos ESS/EER e spread de balcão.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processados_tempr"
RES_TABELA = BASE / "resultados" / "tabela_tempr"
RES_ESTAB = BASE / "resultados" / "estabilidade"
RES_BACKTEST = BASE / "resultados" / "backtesting_out_of_sample"
RES_SENS = BASE / "resultados" / "sensibilidade_regulatoria"
FIG_DIR = BASE / "docs" / "artigo_multiestagio" / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Configuração visual IEEE
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "lines.linewidth": 1.2,
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
})


def plot_pld_spread():
    treino_path = DATA_DIR / "pld_horario_treino_2021_2023.csv"
    teste_path = DATA_DIR / "pld_horario_teste_2024_2025.csv"

    df_tr = pd.read_csv(treino_path)
    df_te = pd.read_csv(teste_path)

    fig, ax = plt.subplots(figsize=(5.0, 2.7))

    sp_tr = np.sort(df_tr["SPREAD_SE_NE"].to_numpy(dtype=float))
    sp_te = np.sort(df_te["SPREAD_SE_NE"].to_numpy(dtype=float))
    ccdf_tr = 1.0 - np.arange(len(sp_tr)) / len(sp_tr)
    ccdf_te = 1.0 - np.arange(len(sp_te)) / len(sp_te)

    ax.step(sp_tr, ccdf_tr, where="post", color="#1f77b4", linewidth=1.3,
            label=f"2021\u20132023 (mean = BRL {sp_tr.mean():.2f}/MWh)")
    ax.step(sp_te, ccdf_te, where="post", color="#d62728", linestyle="-", linewidth=1.3,
            label=f"2024\u20132025 (mean = BRL {sp_te.mean():.2f}/MWh)")
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlim(-60, 610)
    ax.set_ylim(5e-5, 1.3)
    ax.set_xlabel(r"Hourly Locational Spread $\mathrm{PLD}_{\mathrm{SE}} - \mathrm{PLD}_{\mathrm{NE}}$ (BRL/MWh)")
    ax.set_ylabel("Exceedance Prob. (log)")
    ax.set_title("Empirical Hourly Spread Distribution (Complementary CDF)", pad=6)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.0)

    fig.tight_layout()
    out = FIG_DIR / "fig_tempr_pld_spread.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")


def plot_fronteira():
    csv_tab = RES_TABELA / "tabela_experimentos_tempr.csv"
    if not csv_tab.exists():
        return
    df = pd.read_csv(csv_tab)

    mer = df[df["modo"] == "com_revenda"].sort_values("lambda")
    me = df[df["modo"] == "sem_revenda"].sort_values("lambda")
    twos = df[df["modo"] == "dois_estagios"].sort_values("lambda")
    heur = df[df["modo"] == "heuristica_95"].sort_values("lambda")

    fig, ax = plt.subplots(figsize=(5.0, 3.45))

    mer_cvar5 = -mer["cvar_milhoes"]
    me_cvar5 = -me["cvar_milhoes"]
    twos_cvar5 = -twos["cvar_milhoes"]
    heur_cvar5 = -heur["cvar_milhoes"]

    ax.plot(mer_cvar5, mer["er_milhoes"], marker="o", markersize=4.5, color="#1f77b4",
            label="ME-R (Multistage with Resale)", zorder=4)
    ax.plot(me_cvar5, me["er_milhoes"], marker="s", markersize=4.5, color="#2ca02c",
            linestyle="--", label="ME (Multistage Buy-and-Hold)", zorder=3)
    ax.plot(twos_cvar5, twos["er_milhoes"], marker="D", markersize=4.2, color="#9467bd",
            linestyle="-.", label="2S (Two-Stage Mean-CVaR)", zorder=3)
    ax.scatter(heur_cvar5.iloc[0], heur["er_milhoes"].iloc[0], marker="^", color="#d62728",
               s=65, zorder=5, label="HEUR (95% Static Hedge)")

    # Destaque para lambda = 0.5
    p_mer = mer[mer["lambda"] == 0.5].iloc[0]
    p_me = me[me["lambda"] == 0.5].iloc[0]
    d_cvar = (-p_mer["cvar_milhoes"]) - (-p_me["cvar_milhoes"])
    d_fo = p_mer["fo_milhoes"] - p_me["fo_milhoes"]
    cvar5_mer_05 = -p_mer["cvar_milhoes"]

    # Inset zoom para a regiao avessa ao risco (lambda >= 0.1) onde ME-R e ME se separam
    axins = ax.inset_axes([0.10, 0.12, 0.44, 0.44])
    axins.plot(mer_cvar5.iloc[1:], mer["er_milhoes"].iloc[1:], marker="o", markersize=4.5, color="#1f77b4", zorder=4)
    axins.plot(me_cvar5.iloc[1:], me["er_milhoes"].iloc[1:], marker="s", markersize=4.5, color="#2ca02c", linestyle="--", zorder=3)
    axins.set_xlim(22.7, 25.4)
    axins.set_ylim(26.54, 26.84)
    axins.set_xticks([23.0, 23.8, 24.6, 25.4])
    axins.set_yticks([26.6, 26.7, 26.8])
    axins.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    axins.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    axins.tick_params(labelsize=7.0)
    axins.grid(True, alpha=0.35)
    axins.set_title(r"Zoom ($\lambda \in [0.1, 0.95]$)", fontsize=7.5, pad=3)
    axins.annotate(
        rf"$\Delta \mathrm{{CVaR}}_{{0.05}}^{{\mathrm{{profit}}}} = +{d_cvar:.2f}\mathrm{{M}}$" "\n" rf"$\Delta \mathrm{{FO}} = +{d_fo:.2f}\mathrm{{M}}$ ($\lambda=0.5$)",
        xy=(cvar5_mer_05, p_mer["er_milhoes"]),
        xytext=(23.72, 26.74),
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=6.3,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#eef", edgecolor="#88a", alpha=0.95),
        zorder=6,
    )
    ax.indicate_inset_zoom(axins, edgecolor="#666666", linewidth=0.7)

    # Rotulos explicativos sem sobreposicao
    ax.text(heur_cvar5.iloc[0] - 2.0, heur["er_milhoes"].iloc[0] - 0.14, "HEUR",
            fontsize=7.5, color="#a00", fontweight="bold", ha="right", va="top")
    ax.text(twos_cvar5.iloc[1] - 3.0, twos["er_milhoes"].iloc[1] + 0.08, "2S",
            fontsize=7.5, color="#5c3582", fontweight="bold", ha="right", va="bottom")
    ax.text(mer_cvar5.iloc[0] + 3.0, mer["er_milhoes"].iloc[0] + 0.04, r"$\lambda=0$ (Unhedged)",
            fontsize=7.5, ha="left", va="bottom", color="#333")

    y_min = min(mer["er_milhoes"].min(), me["er_milhoes"].min(), heur["er_milhoes"].min()) - 0.35
    y_max = max(mer["er_milhoes"].max(), me["er_milhoes"].max(), heur["er_milhoes"].max()) + 0.35
    ax.set_xlim(-102, 34)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"5% Expected Tail Profit: $\mathrm{CVaR}_{0.05}^{\mathrm{profit}}$ (R\$ Millions)")
    ax.set_ylabel(r"Expected Return: $\mathbb{E}[R]$ (R\$ Millions)")
    ax.set_title("Mean\u2013CVaR Efficient Frontier (Seed 2026 Benchmark)")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=7.5)

    fig.tight_layout()
    out = FIG_DIR / "fig_tempr_fronteira.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")


def plot_estabilidade():
    csv_est = RES_ESTAB / "estabilidade_in_sample.csv"
    if not csv_est.exists():
        return
    df = pd.read_csv(csv_est)

    fig, ax1 = plt.subplots(figsize=(5.0, 2.55))

    color1 = "#1f77b4"
    ax1.set_xlabel(r"Number of Terminal Leaves $|\mathcal{L}|$")
    ax1.set_ylabel("Mean Objective FO (R$ Millions)", color=color1)
    ax1.errorbar(df["n_folhas"], df["media_fo_milhoes"], yerr=df["std_fo_milhoes"],
                 fmt="o-", color=color1, capsize=4, label=r"Mean FO $\pm$ 1 SD (10 seeds)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(df["n_folhas"])
    ax1.set_ylim(19.5, 29.8)
    ax1.grid(True)
    ax1.set_title(r"In-Sample Objective Stability ($\lambda = 0.50$, 10 Seeds)")

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Coefficient of Variation (%)", color=color2)
    ax2.plot(df["n_folhas"], df["cv_fo_pct"], "s--", color=color2, label="CV (%)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 15.0)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=7.5, framealpha=0.95)

    fig.tight_layout()
    out = FIG_DIR / "fig_tempr_estabilidade.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")


def plot_backtest():
    csv_bt = RES_BACKTEST / "backtesting_rolante_mensal_2024_2025.csv"
    if not csv_bt.exists():
        return
    df = pd.read_csv(csv_bt)
    df = df.sort_values(["seed", "estrategia", "mes_absoluto"])
    df["cum_M"] = df.groupby(["seed", "estrategia"])["lucro_liquido_M"].cumsum()

    stats = df.groupby(["estrategia", "mes_absoluto"])["cum_M"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(5.2, 2.95))
    specs = [
        ("ME", "ME (Buy-and-Hold)", "#2ca02c", "-", "s"),
        ("ME-R", "ME-R (With Resale)", "#1f77b4", "-", "o"),
        ("HEUR+R_full", "HEUR+R (full)", "#9467bd", "--", "^"),
        ("HEUR", "HEUR (Static 95%)", "#ff7f0e", "-.", "d"),
        ("DET", "Unhedged (DET)", "#d62728", ":", "x"),
    ]
    for est, label, col, ls, mk in specs:
        sub = stats[stats["estrategia"] == est].sort_values("mes_absoluto")
        x = sub["mes_absoluto"].to_numpy() + 1
        y = sub["mean"].to_numpy()
        sd = sub["std"].fillna(0.0).to_numpy()
        ax.plot(x, y, linestyle=ls, marker=mk, markersize=3.2, markevery=3, color=col, linewidth=1.35, label=label)
        if est in ("ME", "ME-R"):
            ax.fill_between(x, y - sd, y + sd, color=col, alpha=0.10)

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Delivery Month in Out-of-Sample Horizon (1 = Jan 2024, 24 = Dec 2025)")
    ax.set_ylabel("Cumulative Accrual Profit (R$ Millions)")
    ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22, 24])
    ax.set_xlim(1, 24.5)
    ax.set_ylim(-36, 72)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.2, framealpha=0.95, ncol=2)

    fig.tight_layout()
    out = FIG_DIR / "fig_tempr_backtest.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")


def plot_sensibilidade():
    csv_enc = RES_SENS / "sensibilidade_encargos.csv"
    csv_sp = RES_SENS / "sensibilidade_spread.csv"
    if not csv_enc.exists() or not csv_sp.exists():
        return
    df_enc = pd.read_csv(csv_enc)
    df_sp = pd.read_csv(csv_sp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # Subplot 1: Encargos
    for lam in [0.0, 0.5, 0.95]:
        sub = df_enc[df_enc["lambda"] == lam]
        ax1.plot(sub["encargo_mwh"], sub["fo_milhoes"], marker="o", label=rf"$\lambda = {lam}$")
    ax1.set_xlabel(r"Reserve Charge $\tau_{\mathrm{reg}}$ (R\$/MWh)")
    ax1.set_ylabel("Objective Function (R$ Millions)")
    ax1.set_title("(a) System Reserve Charge Sensitivity")
    ax1.grid(True)
    ax1.legend(fontsize=7.5)

    # Subplot 2: Spread
    for lam in [0.0, 0.5]:
        sub = df_sp[df_sp["lambda"] == lam]
        ax2.plot(sub["spread_pct"], sub["fo_milhoes"], marker="s", label=rf"$\lambda = {lam}$")
    ax2.set_xlabel(r"Forward One-Way Half-Spread $\sigma$ (%)")
    ax2.set_ylabel("Objective Function (R$ Millions)")
    ax2.set_title("(b) Forward Market Liquidity Half-Spread")
    ax2.grid(True)
    ax2.legend(fontsize=7.5)

    fig.tight_layout()
    out = FIG_DIR / "fig_tempr_sensibilidade.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")


if __name__ == "__main__":
    plot_pld_spread()
    plot_fronteira()
    plot_estabilidade()
    plot_backtest()
    plot_sensibilidade()
