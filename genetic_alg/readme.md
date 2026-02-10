# Genetic Algorithm for Model Fusion (ResNet + Bioclim/MaxEnt)

This repository implements a Genetic Algorithm (GA) to optimize the
combination of prediction probabilities from two classification models:

-   A Deep Learning model (e.g., ResNet)
-   An Ecological model (e.g., Bioclim / MaxEnt)

Two GA strategies are implemented:

1.  Single global weight (1-weight GA)
2.  Class-specific weights (15-weights GA)

Additionally, a preprocessing script prepares the prediction files used
as input for the GA.

------------------------------------------------------------------------

# Files Overview

## 1. predictions_for_ga.py

Prepares and merges prediction CSV files from:

-   ResNet (image-based classifier)
-   Bioclim / MaxEnt (ecological model)

### What It Does

-   Loads both prediction CSV files
-   Standardizes image IDs
-   Aligns class names
-   Converts class labels to numeric indices (0--14)
-   Merges predictions into a single CSV
-   Saves merged file for GA input

The resulting CSV must contain:

image_id \| 15 probs model1 \| real_class \| 15 probs model2

------------------------------------------------------------------------

## 2. ga_1\_classic_tournament.py

### Single Global Weight Optimization

Finds a single scalar weight w ∈ \[0,1\] that combines:

P_final = w \* P_model1 + (1 - w) \* P_model2

GA Configuration:

-   Tournament selection
-   Arithmetic crossover
-   Gaussian mutation
-   Elitism (top 5 preserved)
-   60 generations
-   Population size: 100
-   30 runs (seeds 42--71)

Outputs:

ga_outputs/1_weight_classic_tournament/
test_results\_`<model>`{=html}.csv val\_`<model>`{=html}.csv
PESOS\_`<model>`{=html}.csv

------------------------------------------------------------------------

## 3. ga_15_classic_tournament.py

### Class-Specific Weight Optimization

Finds 15 independent weights:

P_final_c = w_c \* P_model1_c + (1 - w_c) \* P_model2_c

Each class has its own optimized weight.

Outputs:

ga_outputs/15_weights_classic_tournament/
test_results\_`<model>`{=html}.csv val\_`<model>`{=html}.csv
PESOS\_`<model>`{=html}.csv

------------------------------------------------------------------------

# Required Directory Structure

predictions_for_ga/ `<model_name>`{=html}/ train_predictions.csv
val_predictions.csv test_predictions.csv

Each CSV must contain:

-   image_id
-   15 probabilities from Model 1
-   real_class (numeric 0--14)
-   15 probabilities from Model 2

------------------------------------------------------------------------

# Dependencies

Install:

pip install numpy pandas scikit-learn

------------------------------------------------------------------------

# How to Run

1.  Generate merged predictions:

python predictions_for_ga.py

2.  Run single-weight GA:

python ga_1\_classic_tournament.py

3.  Run 15-weights GA:

python ga_15_classic_tournament.py

------------------------------------------------------------------------

# Evaluation Metric

The GA optimizes classification accuracy based on argmax of combined
probabilities.

------------------------------------------------------------------------

# Research Context

This framework performs late fusion between:

-   A vision-based deep CNN (ResNet)
-   A species distribution model (Bioclim / MaxEnt)

The Genetic Algorithm searches for the best linear combination of
probabilities to maximize classification accuracy.

------------------------------------------------------------------------

# License

For research and academic use.
