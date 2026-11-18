#!/usr/bin/env bash
set -euo pipefail

# Script de compilação do artigo usando tectonic
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

TECTONIC_BIN="$REPO_ROOT/.venv/bin/tectonic"
if [ ! -x "$TECTONIC_BIN" ]; then
    if command -v tectonic >/dev/null 2>&1; then
        TECTONIC_BIN="$(command -v tectonic)"
    else
        echo "Erro: tectonic não encontrado em $TECTONIC_BIN nem no PATH do sistema." >&2
        exit 1
    fi
fi

cd "$SCRIPT_DIR"
echo "Compilando artigo.tex com $TECTONIC_BIN..."
"$TECTONIC_BIN" --keep-logs artigo.tex
echo "Compilação concluída com sucesso: $SCRIPT_DIR/artigo.pdf"
if [ -f "$SCRIPT_DIR/suplemento.tex" ]; then
    echo "Compilando suplemento.tex com $TECTONIC_BIN..."
    "$TECTONIC_BIN" --keep-logs suplemento.tex
    echo "Compilação concluída com sucesso: $SCRIPT_DIR/suplemento.pdf"
fi
