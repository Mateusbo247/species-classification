import os
import shutil
import random
from collections import defaultdict

# Caminhos dos diretórios
input_dir = '/home/user/Final_images/sissgeo-edit'
output_dir = '/home/user/species-classification/dataset'

# Configurações de tamanhos
train_count = 1000
val_count = 300
test_count = 300

def ensure_augmentation_diversity(images, augmented_tags, subset_size):
    """
    Seleciona imagens garantindo que ao menos uma de cada técnica de aumento esteja presente.
    """
    selected = []
    used_tags = set()
    remaining_images = images.copy()

    # Garantir diversidade de aumentos
    for tag, image_list in augmented_tags.items():
        if image_list:
            selected_image = random.choice(image_list)
            if selected_image in remaining_images:  # Verificar antes de remover
                selected.append(selected_image)
                used_tags.add(tag)
                remaining_images.remove(selected_image)

    # Preencher o restante com amostragem aleatória
    while len(selected) < subset_size:
        if not remaining_images:
            break  # Evitar loop infinito se não houver mais imagens disponíveis
        candidate = random.choice(remaining_images)
        if candidate not in selected:
            selected.append(candidate)
            remaining_images.remove(candidate)

    return selected


def split_dataset(input_dir, output_dir, train_count, val_count, test_count):
    """
    Divide o conjunto de dados em treino, validação e teste.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Criar diretórios de saída
        train_path = os.path.join(output_dir, 'train', class_name)
        val_path = os.path.join(output_dir, 'val', class_name)
        test_path = os.path.join(output_dir, 'test', class_name)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Listar imagens da classe
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        # Classificar imagens por tipo de aumento
        augmented_tags = defaultdict(list)
        for img in images:
            if 'aug' in img:
                tag = img.split('_')[1]  # Extrair tipo de aumento
                augmented_tags[tag].append(img)
            else:
                augmented_tags['original'].append(img)

        # Garantir diversidade nos conjuntos
        train_images = ensure_augmentation_diversity(images, augmented_tags, train_count)
        remaining_images = [img for img in images if img not in train_images]
        val_images = ensure_augmentation_diversity(remaining_images, augmented_tags, val_count)
        test_images = [img for img in remaining_images if img not in val_images][:test_count]

        # Verificar se os tamanhos estão corretos
        assert len(train_images) == train_count, f"Erro no tamanho do conjunto de treino para {class_name}"
        assert len(val_images) == val_count, f"Erro no tamanho do conjunto de validação para {class_name}"
        assert len(test_images) == test_count, f"Erro no tamanho do conjunto de teste para {class_name}"

        # Mover imagens para os conjuntos
        for img in train_images:
            shutil.copy(img, train_path)
        for img in val_images:
            shutil.copy(img, val_path)
        for img in test_images:
            shutil.copy(img, test_path)

        print(f"Classe {class_name}: Treino: {len(train_images)}, Validação: {len(val_images)}, Teste: {len(test_images)}")

# Chamada do script
split_dataset(input_dir, output_dir, train_count, val_count, test_count)
