library(modleR)
library(raster)
library(dplyr)

# Configuração inicial
species_list <- c("Species1", "Species2", ..., "Species15") # Substitua pelos nomes reais
occurrence_data <- read.csv("ocurrence_data.csv") # Dados com colunas: species, longitude, latitude
env_layers <- stack(list.files("path_to_env_layers/", pattern = "\\.tif$", full.names = TRUE))

# 1. Preparar os dados de ocorrência
cleaned_data <- setup_sdmdata(species_name = species_list,
                              occurrences = occurrence_data,
                              predictors = env_layers,
                              models_dir = "models",
                              seed = 123,
                              partition_type = "crossvalidation",
                              cv_partitions = 5,
                              buffer_type = "mean",
                              clean_dupl = TRUE,
                              clean_nas = TRUE,
                              clean_uni = TRUE)

# 2. Modelagem para cada espécie
for (species in species_list) {
  do_many(species_name = species,
          predictors = env_layers,
          models_dir = "models",
          seed = 123,
          algorithms = c("maxent", "rf", "svm"),
          project_model = TRUE,
          mask = NULL,
          png_legend = TRUE,
          write_bin_cut = TRUE)
}

# 3. Geração de mapas de distribuição
for (species in species_list) {
  ensemble_model(species_name = species,
                 occurrences = occurrence_data,
                 models_dir = "models",
                 which_models = c("maxent", "rf", "svm"),
                 consensus_level = 0.5,
                 stat = "mean",
                 png_legend = TRUE)
}

# 4. Avaliação e criação da tabela de probabilidade
all_eval <- list()
for (species in species_list) {
  eval <- get_evaluations(species_name = species, models_dir = "models")
  eval_df <- eval %>% 
    dplyr::mutate(species = species) %>% 
    dplyr::select(species, algorithm, auc, sensitivity, specificity)
  all_eval <- rbind(all_eval, eval_df)
}

# Exportar a tabela de probabilidade
write.csv(all_eval, "model_evaluation.csv", row.names = FALSE)

# 5. Unir as predições em um único raster para visualizar todas as espécies
distribution_stack <- stack(list.files("models", pattern = "ensemble_mean\\.tif$", full.names = TRUE))
names(distribution_stack) <- species_list
writeRaster(distribution_stack, "species_distribution.tif", overwrite = TRUE)
