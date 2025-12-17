import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

set_seed(42)

# =====================================================
# Configurações gerais
# =====================================================
data_dir = 'dataset'
batch_size = 32
num_epochs = 60
num_classes = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_stop_patience = 10
model_save_path = 'best_model_resnet152.pt'
os.makedirs("predictions", exist_ok=True)

# =====================================================
# Transformações de imagem
# =====================================================
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# =====================================================
# Datasets e Dataloaders
# =====================================================
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform[x])
    for x in ['train', 'val', 'test']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'))
    for x in ['train', 'val', 'test']
}

# =====================================================
# Modelo ResNet-152
# =====================================================
model = models.resnet152(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# =====================================================
# Otimizador e função de perda
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# =====================================================
# Função de treinamento com early stopping
# =====================================================
def train_model():
    best_acc = 0.0
    best_model_wts = model.state_dict()
    history = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_stats = {'epoch': epoch+1}

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            epoch_stats[f'{phase}_loss'] = epoch_loss
            epoch_stats[f'{phase}_acc'] = epoch_acc.item()

            # Early stopping com validação
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), model_save_path)
                    print(" Novo melhor modelo salvo!")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        history.append(epoch_stats)

        if epochs_no_improve >= early_stop_patience:
            print(f"\n Early stopping após {early_stop_patience} épocas sem melhoria.")
            break

    model.load_state_dict(best_model_wts)

    df = pd.DataFrame(history)
    df.to_csv('training_history.csv', index=False)
    print(" Histórico de treino salvo em 'training_history.csv'.")

# =====================================================
# Avaliação no conjunto de teste
# =====================================================
def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = image_datasets['test'].classes
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("classification_report_test.csv")
    print(" Relatório de teste salvo em 'classification_report_test.csv'.")

    print("\nResumo do Teste:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# =====================================================
# Geração de CSV com probabilidades (PyTorch)
# =====================================================
def generate_predictions_csv(dataloader, model, dataset_name, output_dir="predictions"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_probs = []
    all_image_paths = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

            # Caminhos corretos das imagens deste batch
            batch_paths = [
                dataloader.dataset.samples[i + batch_idx * dataloader.batch_size][0]
                for i in range(len(inputs))
            ]
            all_image_paths.extend(batch_paths)

    image_ids = [os.path.basename(p) for p in all_image_paths]
    class_labels = image_datasets[dataset_name].classes

    df = pd.DataFrame(all_probs, columns=class_labels)
    df.insert(0, "image_id", image_ids)
    df["true_label"] = [class_labels[i] for i in all_labels]

    csv_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f" CSV de previsões salvo: {csv_path}")

# =====================================================
# Matriz de confusão
# =====================================================
def plot_confusion_matrix(dataloader, model):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = dataloader.dataset.classes
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

# =====================================================
# Execução principal
# =====================================================
if __name__ == "__main__":
    train_model()
    evaluate_model()

    print("\n Gerando arquivos CSV com probabilidades...")
    generate_predictions_csv(dataloaders["val"], model, "val")
    generate_predictions_csv(dataloaders["test"], model, "test")

    print("\n Gerando matriz de confusão do conjunto de teste...")
    plot_confusion_matrix(dataloaders["test"], model)
