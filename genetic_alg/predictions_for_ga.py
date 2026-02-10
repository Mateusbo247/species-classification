
import os
import pandas as pd
import unicodedata
import re

BATCH = 1
MAXENT_DIR =  f'/home/antonio/mateus_results/maxent_vselection/train_results/bioclim/bioclim_train_global.csv'
RESNET_DIR = f'/home/antonio/mateus_results/ResNet50/train_predictions_label.csv'
# Ler os dois CSVs
df_maxent = pd.read_csv(MAXENT_DIR)  # contém coluna 'id' (números)
df_resnet = pd.read_csv(RESNET_DIR)  # contém coluna 'image_id' (ex: '123.jpg')
colunas_maxent = ['Registro..Identificador', 'species',  'anfibio', 'ave-de-rapina', 
	'cachorro', 'capivara', 'cobra', 'cutia', 'gamba', 'lagarto', 'macaco', 'mico', 
	'morcego', 'preguica', 'quati', 'tamandua', 'tartaruga'                                
   	]
df_maxent = df_maxent[colunas_maxent]

df_maxent = df_maxent.rename(columns={
    "Registro..Identificador": "image_id",
    "species": "real_class"
})


df_resnet["image_id"] = df_resnet["image_id"].str.split("-").str[0]

df_maxent["image_id"] = df_maxent["image_id"].astype(int)
df_resnet["image_id"] = df_resnet["image_id"].astype(int)

df_resnet = df_resnet.rename(columns={
    "true_label": "real_class"
})

print(f'Maxent: {len(df_maxent)}')
print(f'Resnet: {len(df_resnet)}')

df_maxent = df_maxent.drop_duplicates(
    subset=["image_id", "real_class"],
    keep="first"
)

print(f'Maxent: {len(df_maxent)}')
print(f'Resnet: {len(df_resnet)}')
print(df_maxent.columns.tolist())
print(df_resnet.columns.tolist())

df_maxent = df_maxent.drop(columns=["real_class"])

# Fazer o left join com base na coluna 'id'
df_merged = pd.merge(df_resnet, df_maxent, on=['image_id'], how='left')
classes_ordenadas = ['anfibio', 'ave-de-rapina', 
	'cachorro', 'capivara', 'cobra', 'cutia', 'gamba', 'lagarto', 'macaco', 'mico', 
	'morcego', 'preguica', 'quati', 'tamandua', 'tartaruga']
mapa_classes = {nome: i for i, nome in enumerate(classes_ordenadas)}
df_merged['real_class'] = df_merged['real_class'].map(mapa_classes)
df_merged = df_merged.dropna(subset=["real_class"])
print(f'Merged: {len(df_merged)}')
print(df_merged.columns.tolist())
print(df_merged.head())
# Caminho do arquivo
caminho_arquivo = f'/home/antonio/mateus_results/predictions_for_ga/resnet50_bioclim/train_predictions.csv'
# Extrai o diretório do caminho
diretorio = os.path.dirname(caminho_arquivo)
# Cria o diretório se não existir
os.makedirs(diretorio, exist_ok=True)
# Salvar o resultado
df_merged.to_csv(caminho_arquivo, index=False)


