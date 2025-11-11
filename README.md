# Animal species-classification: Integration of Deep-Learning and Species Distribution Models for Classification of Animal Species of the Brazilian Fauna

This project seeks to bring a combination between image classification models and species distribution models, focusing on the classification of animal species of the Brazilian fauna.

The automated classification of animals from photos is important in ecology and conservation biology for organizing and understanding the immense diversity of species, as well as facilitating effective conservation and management practices. It is equally important for disease surveillance systems, allowing prompt detection of anomalies in species distributions and boosting citizen-scientist platforms by making user-reported data more accurate and convenient. Image classification uses photos and can also rely on the geographical locations of animals to improve performance. While image classification models have difficulties in classifying low-quality images, unbalanced datasets, and with a small number of images, species distribution models have difficulty in classifying species that coexist in a given region. The present work presents an integration of image classification models based on deep neural networks with species distribution models. It is applied to a real-world dataset comprising fifteen classes of animals from the Brazilian fauna obtained from Fiocruz's citizen-scientist Wildlife Health Information System (SISS-Geo). The SISS-Geo photos portray the reality of animals in their environments, with varying quality, and pose numerous difficulties for classification.

The project structure is as follows:

 - class_id: Given a CSV file containing rows of the form: id_register,id_animal,class_index; and a bunch of JPEG filenames as id_register-image_number.jpg, this will assign the class index to each of these filenames in the form - id_register-class_index-image_number.jpg
 - cnn_model: Image classification model; The ResNet-50 network is used to classify images of animals from the Brazilian fauna.
 - data_augmentation: From the images, the data augmentation process is carried out to treat and balance the set of images available for the execution of the deep neural network. The operations of mirroring, zooming, rotating and tiling the images are carried out. Furthermore, the dimensions of the images are standardized.
 - genetic_alg: Genetic algorithm to combine image classification and species distribution models.
 - sdm_model: Species distribution model, used to estimate the existence of animals according to their geographic location.
 - sissgeo_dataset: Dataset with records of Brazilian fauna animals used for classification.

# Model Execution
- Data preprocessing: Organization and separation of data by class; Application of data augmentation to balance the original set; Organization of data for training, validation and testing.
- Model Training: ResNet and MaxEnt Training.
- Preparing data for analysis: Organizing data into tables that contain all predicted probabilities for ResNet and MaxEnt.
- Execution of the Genetic Algorithm: Training of the Genetic Algorithm to generate 1 and 15 parameters for combining ResNet and MaxEnt.
- Analysis of results: Analysis of metrics to compare the results obtained by all trained models.


==========================================
CLASSIFICATION AND MODELING PIPELINE
==========================================

------------------------------------------
1. PREPROCESSING
------------------------------------------

• Class Identification
  - Script: assign_class_index1.py
  - Purpose: assigns each image to its corresponding record and class.
  - Input: image directory and metadata file (e.g., labels.csv)
  - Output: dataset with numeric labels assigned to each class.

• Dataset Split
  - Script: dataset-division2.py
  - Purpose: splits the dataset into training, validation, and test subsets.
  - Parameters: adjustable ratios (e.g., 70% train, 15% validation, 15% test).
  - Output: organized folders /train, /val, and /test.

• Data Augmentation
  - Script: augmentation.py
  - Purpose: generates new samples from original images by applying
             transformations (rotation, flipping, zoom, brightness, etc.).
  - Output: expanded dataset for improved model generalization.

------------------------------------------
2. MODEL EXECUTION
------------------------------------------

• ResNet-50
  - Script: resnet-50-csv.py
  - Purpose: trains a ResNet-50 neural network and logs results in CSV format
             (accuracy, loss, confusion matrix).
  - Input: preprocessed dataset.
  - Output: trained weights and performance metrics.

• ResNet-152
  - Script: resnet-152.py
  - Purpose: trains a deeper ResNet-152 network for comparison with ResNet-50.
  - Output: model weights and performance metrics.

• MaxEnt and BIOCLIM
  - Script: max-bio.R
  - Purpose: runs ecological niche models using MaxEnt and BIOCLIM algorithms
             via the ModleR package in R.
  - Output: predictive maps and evaluation metrics (AUC, TSS, etc.).

------------------------------------------
3. MODEL COMBINATION
------------------------------------------

• Genetic Algorithm
  - Script: alg_gen15.py
  - Purpose: combines predictions from all models (ResNet-50, ResNet-152, MaxEnt, BIOCLIM)
             through optimization with a genetic algorithm.
  - Objective: find the optimal weighting among models to maximize global accuracy.
  - Output: optimized hybrid model and consolidated performance metrics.


END OF PROCEDURE
------------------------------------------

