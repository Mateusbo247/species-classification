import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configurações
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60
NUM_CLASSES = 15
OUTPUT_DIR = "./output_csv"  # Caminho para salvar os CSVs

# Caminhos para os conjuntos de dados
TRAIN_DIR = "./dataset/train"
VAL_DIR = "./dataset/val"
TEST_DIR = "./dataset/test"

# Criar pasta para os CSVs, se necessário
os.makedirs(OUTPUT_DIR, exist_ok=True)

CALLBACKS = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1, min_delta=0.0001)
]

def preprocess_resnet(img):
    return preprocess_input(img)

# Geradores de dados
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_resnet, 
                                   rotation_range=20, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_resnet)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Carregar a ResNet-50 sem a última camada (headless)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Congelar camadas do modelo base
base_model.trainable = False

# Construir o modelo
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=CALLBACKS
)

# Salvar o modelo treinado
model.save("resnet50_model.h5")

# Avaliação no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Função para gerar previsões e salvar CSV
def generate_predictions_csv(generator, model, output_path, dataset_name):
    """
    Gera previsões para um conjunto de dados e salva um arquivo CSV com as probabilidades.
    """
    predictions = model.predict(generator, verbose=1)
    image_ids = [os.path.basename(path) for path in generator.filenames]
    class_labels = list(generator.class_indices.keys())

    # Criar DataFrame
    df = pd.DataFrame(predictions, columns=class_labels)
    df.insert(0, 'image_id', image_ids)

    # Salvar CSV
    csv_path = os.path.join(output_path, f"{dataset_name}_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"{dataset_name.capitalize()} CSV salvo em: {csv_path}")

# Geração de arquivos CSV para cada conjunto
generate_predictions_csv(train_generator, model, OUTPUT_DIR, "train")
generate_predictions_csv(val_generator, model, OUTPUT_DIR, "validation")
generate_predictions_csv(test_generator, model, OUTPUT_DIR, "test")

# Predições no conjunto de teste
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Relatório de classificação
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Acurácia
accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {accuracy}")

