import os
import cv2
import json
import torch
import shutil
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# CONFIGURAÇÕES
DATASET_DIR = 'dataset'  # dataset original organizado por classe
YOLO_DATASET_DIR = 'yolo_dataset'
YOLO_MODEL = 'yolov8n.pt'
EPOCHS = 20
IMG_SIZE = 640
NUM_CLASSES = 15
CROP_OUTPUT_DIR = 'crops_dataset'
RESULT_CSV = 'resultados_topk.csv'

# ETAPA 1: Gerar labels para YOLO com bounding box cobrindo a imagem inteira
def generate_yolo_labels(dataset_path, output_path):
    class_names = sorted(os.listdir(os.path.join(dataset_path, 'train')))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        for class_name in os.listdir(split_path):
            class_dir = os.path.join(split_path, class_name)
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                # bbox cobrindo toda a imagem
                label_id = class_to_id[class_name]
                label_line = f"{label_id} 0.5 0.5 1.0 1.0\n"

                # salvar imagem e label
                new_img_dir = os.path.join(output_path, split, 'images')
                new_label_dir = os.path.join(output_path, split, 'labels')
                os.makedirs(new_img_dir, exist_ok=True)
                os.makedirs(new_label_dir, exist_ok=True)

                new_img_path = os.path.join(new_img_dir, f"{class_name}_{img_name}")
                new_label_path = os.path.join(new_label_dir, f"{class_name}_{os.path.splitext(img_name)[0]}.txt")

                cv2.imwrite(new_img_path, img)
                with open(new_label_path, 'w') as f:
                    f.write(label_line)

    return class_to_id

# ETAPA 2: Criar arquivo YAML de configuração do dataset
def create_data_yaml(class_to_id, yaml_path, dataset_path):
    with open(yaml_path, "w") as f:
        f.write(f"""path: {dataset_path}
train: train/images
val: val/images
test: test/images
nc: {len(class_to_id)}
names: {list(class_to_id.keys())}
""")

# ETAPA 3: Avaliar top-k e salvar em CSV
def evaluate_topk(model, test_dir, class_to_id, k_values=[1, 3, 5]):
    rows = []
    reverse_class_map = {v: k for k, v in class_to_id.items()}

    for img_name in tqdm(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, img_name)
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(img_path)
        results = model(img)[0]
        if len(results.boxes.cls) == 0:
            predicted_ids = []
        else:
            probs = results.probs
            predicted_ids = torch.topk(probs, k=max(k_values), dim=0).indices.cpu().tolist()

        true_class = img_name.split("_")[0]  # nome da classe original na imagem
        true_id = class_to_id[true_class]

        row = {"img": img_name, "true_class": true_class}
        for k in k_values:
            topk = predicted_ids[:k]
            row[f"top{k}_correct"] = int(true_id in topk)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False)
    print("\nResumo de métricas:")
    for k in k_values:
        acc = df[f"top{k}_correct"].mean()
        print(f"Top-{k} Accuracy: {acc:.2%}")

# ETAPA 4: Recortar objetos detectados e salvar por classe
def crop_detected_objects(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        results = model(img)[0]

        for i, box in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped = img[y1:y2, x1:x2]
            class_name = model.names[int(cls)]
            class_folder = os.path.join(output_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            crop_filename = f"{os.path.splitext(img_name)[0]}_crop{i}.jpg"
            cv2.imwrite(os.path.join(class_folder, crop_filename), cropped)

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    print("🔧 Gerando labels YOLO...")
    class_map = generate_yolo_labels(DATASET_DIR, YOLO_DATASET_DIR)
    with open("class_mapping.json", "w") as f:
        json.dump(class_map, f)

    print("📝 Criando arquivo YAML...")
    create_data_yaml(class_map, "yolo_dataset.yaml", YOLO_DATASET_DIR)

    print("🚀 Treinando YOLO...")
    model = YOLO(YOLO_MODEL)
    model.train(data="yolo_dataset.yaml", epochs=EPOCHS, imgsz=IMG_SIZE)

    print("📈 Avaliando Top-K...")
    model = YOLO("runs/detect/train/weights/best.pt")
    evaluate_topk(model, os.path.join(YOLO_DATASET_DIR, "test/images"), class_map)

    print("✂️ Recortando objetos detectados...")
    crop_detected_objects(model, os.path.join(YOLO_DATASET_DIR, "test/images"), CROP_OUTPUT_DIR)

    print("✅ Concluído!")

