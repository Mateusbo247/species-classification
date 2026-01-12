import pandas as pd

# Install odfpy if not already installed
!pip install odfpy

# ===============================
# 1. ARQUIVOS DE ENTRADA
# ===============================

arquivo_ods = "registros_siss-geo_20200114.ods"
arquivo_csv_tipo = "id_registro-id_animal-id_tipo.csv"   # CSV da imagem (id_registro, id_tipo)

# ===============================
# 2. LER ODS E EXTRAIR COORDENADAS
# ===============================

df_geo = pd.read_excel(arquivo_ods, engine="odf")

df_geo = df_geo[
    [
        "Registro: Identificador",
        "Registro: Latitude",
        "Registro: Longitude"
    ]
].rename(columns={
    "Registro: Identificador": "id",
    "Registro: Latitude": "lat",
    "Registro: Longitude": "lon"
})

# Remover linhas sem coordenadas
df_geo = df_geo.dropna(subset=["lat", "lon"])

# Garantir tipo inteiro do ID
df_geo["id"] = df_geo["id"].astype(int)

# ===============================
# 3. LER CSV DE TIPOS
# ===============================

df_tipo = pd.read_csv(arquivo_csv_tipo)

df_tipo = df_tipo.rename(columns={
    "id_registro": "id",
    "id_tipo": "tipo"
})

df_tipo["id"] = df_tipo["id"].astype(int)

# ===============================
# 4. MERGE (JUNTAR TIPO AO GEO)
# ===============================

df_final = df_geo.merge(
    df_tipo[["id", "tipo"]],
    on="id",
    how="left"
)

# ===============================
# 5. SALVAR RESULTADO FINAL
# ===============================

df_final.to_csv("registros_geo_com_tipo.csv", index=False)

print("Arquivo gerado: registros_geo_com_tipo.csv")
print(df_final.head())
