import os
import re
import shutil
from sklearn.model_selection import train_test_split

# Diretório de origem (cada subpasta = classe)
origem = "sissgeo-2"
# Diretório de destino
destino = "dataset"

# Razões para split
train_size = 0.7
val_size = 0.15
test_size = 0.15

# Regex para extrair o código inicial (ajuste conforme padrão dos nomes)
# Exemplo: 123.jpg, 123_aug1.png -> código inicial = 123
def get_codigo(filename):
    match = re.match(r"(\d+)", filename)  # pega só os dígitos iniciais
    return match.group(1) if match else filename

# Limpa destino se já existir
if os.path.exists(destino):
    shutil.rmtree(destino)
os.makedirs(destino)

# Percorre cada classe
for classe in os.listdir(origem):
    pasta_classe = os.path.join(origem, classe)
    if not os.path.isdir(pasta_classe):
        continue

    # Agrupa imagens por código inicial
    grupos = {}
    for img in os.listdir(pasta_classe):
        codigo = get_codigo(img)
        grupos.setdefault(codigo, []).append(img)

    codigos = list(grupos.keys())

    # Split estratificado em códigos (não em imagens)
    train_codigos, temp_codigos = train_test_split(
        codigos, train_size=train_size, random_state=42
    )
    val_codigos, test_codigos = train_test_split(
        temp_codigos, test_size=test_size/(test_size+val_size), random_state=42
    )

    splits = {
        "train": train_codigos,
        "val": val_codigos,
        "test": test_codigos,
    }

    # Copia para destino
    for split, codigos_set in splits.items():
        destino_classe = os.path.join(destino, split, classe)
        os.makedirs(destino_classe, exist_ok=True)
        for codigo in codigos_set:
            for img in grupos[codigo]:
                src = os.path.join(pasta_classe, img)
                dst = os.path.join(destino_classe, img)
                shutil.copy2(src, dst)

print("Divisão concluída com sucesso!")
