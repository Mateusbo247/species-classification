############################################################
# Script: modelr_bioclim_maxent.R
############################################################

# --- 1. Instalação e carregamento de pacotes ----------------

# Instalar pacotes básicos (caso não tenha)
install.packages(c("remotes", "raster", "sp", "dismo", "sf"))

# Instalar ModleR a partir do GitHub
if (!require("modelr")) remotes::install_github("Model-R/modelr")

# Carregar pacotes
library(modelr)
library(raster)
library(sp)
library(dismo)
library(sf)

cat("\n✅ Pacotes carregados com sucesso.\n")

# --- 2. Baixar camadas bioclimáticas (WorldClim) -------------

cat("\n⬇️  Baixando camadas bioclimáticas do WorldClim...\n")

# Baixa as 19 variáveis bioclimáticas (resolução 10 min)
env_layers <- getData("worldclim", var = "bio", res = 10)

# Visualizar uma camada (opcional)
plot(env_layers[[1]], main = "BIO1 - Temperatura média anual")

cat("\n✅ Camadas bioclimáticas carregadas.\n")

# --- 3. Gerar dados de ocorrência simulados -----------------

cat("\n🧬 Gerando pontos de ocorrência fictícios...\n")

set.seed(123)
lon <- runif(30, min = -70, max = -40)
lat <- runif(30, min = -25, max = 0)
species <- rep("Especie_demo", length(lon))
occs <- data.frame(species, lon, lat)

# Visualizar no mapa
plot(env_layers[[1]], main = "Ocorrências simuladas")
points(occs$lon, occs$lat, col = "red", pch = 19)

cat("\n✅ Dados de ocorrência prontos.\n")

# --- 4. Organizar estrutura de diretórios --------------------

cat("\n📁 Criando estrutura de diretórios...\n")

dir.create("modelr_data", showWarnings = FALSE)
dir.create("modelr_data/occurrences", showWarnings = FALSE)
dir.create("modelr_data/variables", showWarnings = FALSE)

# Salvar ocorrências
write.csv(occs, "modelr_data/occurrences/ocorrencias.csv", row.names = FALSE)

# Salvar camadas bioclimáticas (em formato GeoTIFF)
for (i in 1:nlayers(env_layers)) {
  writeRaster(env_layers[[i]],
              filename = paste0("modelr_data/variables/bio", i, ".tif"),
              format = "GTiff",
              overwrite = TRUE)
}

cat("\n✅ Estrutura e arquivos salvos.\n")

# --- 5. Rodar os modelos (BIOCLIM e MAXENT) ------------------

cat("\n🚀 Rodando modelagem com BIOCLIM e MAXENT...\n")

do_many(
  species_name = "Especie_demo",
  occurrences = "modelr_data/occurrences/ocorrencias.csv",
  variables_dir = "modelr_data/variables",
  algorithms = c("bioclim", "maxent"),
  partition_type = "crossvalidation",
  n_partitions = 3,
  models_dir = "modelr_results",
  project_model = TRUE
)

cat("\n✅ Modelagem concluída.\n")

# --- 6. Visualizar os resultados -----------------------------

cat("\n🖼️  Visualizando mapas preditivos...\n")

# BIOCLIM
bio_map <- raster("modelr_results/Especie_demo/bioclim/projections/bioclim_current.tif")
plot(bio_map, main = "BIOCLIM - Distribuição Potencial")
points(occs$lon, occs$lat, col = "red", pch = 19)

# MAXENT
max_map <- raster("modelr_results/Especie_demo/maxent/projections/maxent_current.tif")
plot(max_map, main = "MAXENT - Distribuição Potencial")
points(occs$lon, occs$lat, col = "red", pch = 19)

cat("\n✅ Mapas plotados com sucesso.\n")

# --- 7. Avaliar os modelos (opcional) -----------------------

cat("\n📊 Calculando métricas de avaliação...\n")

evaluate_many(models_dir = "modelr_results",
              species_name = "Especie_demo")

cat("\n✅ Avaliação concluída. Resultados salvos em modelr_results/.\n")

############################################################
# Fim do script
############################################################
