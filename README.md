# Contract resale in the Brazilian electricity market

Code for “Contract Resale Flexibility Under Market and Regulatory Constraints in the Brazilian Electricity Market”. The article and supplement are in `nyx/docs/artigo_multiestagio/`.

Install Python dependencies with `pip install -r requirements.txt`. Prepare CCEE price series with `python nyx/data_processing/preparar_dados_tempr.py`. Run the main comparison with `python nyx/scripts/gerar_tabela_oficial_tempr.py`, the rolling backtest with `python nyx/scripts/backtest_rolante_tempr.py`, and figures with `python nyx/scripts/gerar_figuras_tempr.py`. Some figures require results of sensitivity and scenario-stability runs. Compile the manuscript with Tectonic using `bash nyx/docs/artigo_multiestagio/compilar.sh`.

The code is provided under the MIT License. CCEE data may be subject to their original terms.
