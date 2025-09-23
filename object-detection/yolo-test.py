import os
import cv2
import json
import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# CONFIGURAÇÕES
YOLO_MODEL_PATH = 'runs/detect/train2/weights/best.pt'  # ajuste se necessário
YOLO_DATASET_DIR = 'yolo_dataset'
RESULT_CSV = 'resultados_topk.csv'
K_VALUES = [1, 3, 5]

# Função de avaliação
def evaluate_topk(model, test_dir, class_to_id, k_values=[1, 3, 5]):
    rows = []
    reverse_class_map = {v: k for k, v in class_to_id.items()}

    for img_name in tqdm(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, img_name)
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Inferência com probabilidade
        results = model.predict(img, conf=0.001, save_conf=True, verbose=False)[0]
        probs = results.probs

        if probs is None:
            predicted_ids = []
        else:
            predicted_ids = torch.topk(probs, k=max(k_values), dim=0).indices.cpu().tolist()

        true_class = img_name.split("_")[0]
        true_id = class_to_id.get(true_class, -1)

        row = {"img": img_name, "true_class": true_class}
        for k in k_values:
            topk = predicted_ids[:k]
            row[f"top{k}_correct"] = int(true_id in topk)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False)

    print("\n📊 Resumo de métricas:")
    for k in k_values:
        acc = df[f"top{k}_correct"].mean()
        print(f"Top-{k} Accuracy: {acc:.2%}")

# Execução principal
if __name__ == "__main__":
    print("📦 Carregando modelo treinado...")
    model = YOLO(YOLO_MODEL_PATH)

    print("🔁 Carregando mapeamento de classes...")
    with open("class_mapping.json") as f:
        class_map = json.load(f)

    print("🧪 Avaliando conjunto de teste...")
    test_images_dir = os.path.join(YOLO_DATASET_DIR, "test/images")
    evaluate_topk(model, test_images_dir, class_map, k_values=K_VALUES)

    print(f"\n✅ Resultados salvos em: {RESULT_CSV}")

