###########################################
# Pacotes necessários
library(rJava)
library(raster)
library(modleR)
library(dplyr)
library(maps)
library(maptools)
library(rgeos)

###########################################
# Leitura dos dados
especies <- read.table("Animais/s_mico.csv", header = TRUE, sep = ",")
coord1sp <- especies

# Pasta de saída
test_folder <- "~/acm_article-pseudo"

###########################################
# Preditores ambientais
predictor <- example_vars

# Transformar coordenadas em objeto espacial
pts <- SpatialPoints(coord1sp[, c(2, 3)])

###########################################
# Setup dos dados para SDM
m <- setup_sdmdata(
  species_name = especies[1,1],
  occurrences = coord1sp[, -1], 
  predictors = example_vars, 
  models_dir = test_folder, 
  real_absences = NULL,
  buffer_type = "mean", 
  dist_buf = NULL, 
  env_filter = TRUE,
  env_distance = "mindist", 
  buffer_shape = NULL, 
  min_env_dist = 0.2,
  min_geog_dist = NULL, 
  write_buffer = FALSE, 
  seed = NULL,
  clean_dupl = FALSE, 
  clean_nas = FALSE, 
  clean_uni = FALSE,
  geo_filt = FALSE, 
  geo_filt_dist = NULL, 
  select_variables = FALSE,
  cutoff = 0.7, 
  sample_proportion = 0.5, 
  png_sdmdata = TRUE,
  n_back = 500,
  partition_type = "bootstrap",
  boot_n = 5,
  boot_proportion = 0.7
)

###########################################
# Rodando o modelo Maxnet
sp_maxnet <- do_any(
  species_name = especies[1,1],
  algorithm = "maxnet",
  predictors = example_vars,
  models_dir = test_folder,
  png_partitions = TRUE,
  write_bin_cut = FALSE,
  equalize = TRUE
)

###########################################
# >>> NOVO BLOCO: EXTRAÇÃO DAS PROBABILIDADES
###########################################

# Localizar o raster de predição
pred_path <- list.files(
  path = file.path(test_folder, especies[1,1], "maxnet", "prediction"),
  pattern = "\\.tif$",
  full.names = TRUE
)

# Carregar raster de probabilidade
pred_raster <- raster(pred_path[1])

# Extrair probabilidades para cada registro
coords <- coord1sp[, c(2, 3)]
probabilidades <- raster::extract(pred_raster, coords)

# Criar tabela final
resultado <- data.frame(
  especie = coord1sp[, 1],
  longitude = coord1sp[, 2],
  latitude = coord1sp[, 3],
  probabilidade = probabilidades
)

###########################################
# Salvar CSV
write.csv(
  resultado,
  file = file.path(test_folder, "probabilidades_maxnet.csv"),
  row.names = FALSE
)

###########################################
# Visualização rápida
head(resultado)
