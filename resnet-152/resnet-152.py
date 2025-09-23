import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Configurações
data_dir = 'dataset-edit'
batch_size = 32
num_epochs = 60
num_classes = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_stop_patience = 10
model_save_path = 'best_model_resnet152.pt'

# Transformações
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

# Datasets e Dataloaders
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform[x])
    for x in ['train', 'val', 'test']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'))
    for x in ['train', 'val', 'test']
}

# Modelo ResNet-152 com fine-tuning na última camada
model = models.resnet152(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Otimizador e função de perda
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# Função de treinamento com early stopping e salvamento
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

            # Early stopping baseado na validação
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), model_save_path)
                    print("🔽 Modelo salvo!")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        history.append(epoch_stats)

        if epochs_no_improve >= early_stop_patience:
            print(f"\n⏹️ Early stopping após {early_stop_patience} épocas sem melhoria.")
            break

    # Restaurar melhor modelo
    model.load_state_dict(best_model_wts)

    # Salvar histórico em CSV
    df = pd.DataFrame(history)
    df.to_csv('training_history.csv', index=False)
    print("📄 Histórico de treino salvo em 'training_history.csv'.")

# Avaliação no teste
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

    # Salvar como CSV
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("classification_report_test.csv")
    print("📄 Relatório de teste salvo em 'classification_report_test.csv'.")

    # Exibir resumo
    print("\nResumo do Teste:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Execução
if __name__ == "__main__":
    train_model()
    evaluate_model()
