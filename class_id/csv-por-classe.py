import pandas as pd
import os

# Caminho do CSV
csv_path = "s_teste.csv"   

# Ler o CSV
df = pd.read_csv(csv_path)

# Coluna que define a separação
coluna_sp = "sp"

# Pasta de saída
output_dir = "csv_por_sp"
os.makedirs(output_dir, exist_ok=True)

# Separar e salvar
for sp, df_sp in df.groupby(coluna_sp):
    nome_arquivo = f"{output_dir}/sp_{sp}.csv"
    df_sp.to_csv(nome_arquivo, index=False)
    print(f"Arquivo salvo: {nome_arquivo}")
