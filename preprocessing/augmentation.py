import os
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

# Diretório principal contendo as pastas das classes
input_dir = '/home/user/Final_images/sissgeo'
output_dir = '/home/user/Final_images/sissgeo-edit'
target_count = 10000

# Funções de aumento de dados
def augment_image(image):
    """
    Aplica uma sequência aleatória de aumentos na imagem.
    """
    # Conversão para manter a qualidade
    image = image.convert("RGB")

    # Rotação aleatória
    angle = random.randint(-30, 30)
    image = image.rotate(angle)

    # Espelhamento horizontal
    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    # Zoom
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.ANTIALIAS)
        image = image.crop((0, 0, w, h))

    # Ajuste de brilho
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    # Adicionando ruído
    if random.random() > 0.5:
        np_image = np.array(image)
        noise = np.random.randint(0, 50, np_image.shape, dtype='uint8')
        np_image = np.clip(np_image + noise, 0, 255)
        image = Image.fromarray(np_image)

    return image

# Função principal
def augment_dataset(input_dir, output_dir, target_count):
    """
    Aumenta o conjunto de dados para atingir o número de imagens desejado por classe.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_count = len(images)

        print(f"Classe: {class_name} | Imagens originais: {image_count}")

        # Copiar imagens originais para o diretório de saída
        for image_path in images:
            image_name = os.path.basename(image_path)
            image = Image.open(image_path)
            image.save(os.path.join(output_class_path, image_name))

        # Gerar imagens aumentadas até atingir o limite desejado
        while image_count < target_count:
            image_path = random.choice(images)
            image = Image.open(image_path)

            augmented_image = augment_image(image)
            augmented_image_name = f"aug_{image_count}.jpg"
            augmented_image.save(os.path.join(output_class_path, augmented_image_name))
            image_count += 1

        print(f"Classe: {class_name} | Total final de imagens: {image_count}")

# Chamada do script
augment_dataset(input_dir, output_dir, target_count)
