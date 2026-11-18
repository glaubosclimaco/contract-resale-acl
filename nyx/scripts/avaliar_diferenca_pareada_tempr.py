"""
Script oficial para avaliacao pareada in-sample (ME-R menos ME) nas 10 sementes
independentes da arvore CCEE (128 folhas) para todos os niveis de aversao ao risco lambda.
Salva os resultados detalhados e consolidados em nyx/resultados/estabilidade/.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
for sub in ("config", "optimization", "optimization/models", "data_processing", "utils"):
    sys.path.append(str(BASE / sub))

from estabilidade_cenarios import avaliar_diferenca_pareada_sementes


if __name__ == "__main__":
    avaliar_diferenca_pareada_sementes()
