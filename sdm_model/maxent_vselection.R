library(modleR)
library(geodata)
library(terra)
library(raster)
library(dplyr)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(stringr)


sanitize_maxent <- function(x) {
  gsub('[>!´<#?&/\\\\\\.]', "", x)
}



# Ler ocorrências
occs <- read.csv(paste0('/home/antonio/replicacao_mateus/dataset/train_registros_species.csv'))

# Renomear colunas de coordenadas
occs <- occs %>%
  dplyr::rename(
	lon = 'Registro..Longitude',
	lat = 'Registro..Latitude'
  )

occs <- occs %>%
  filter(
    !is.na(lat),
    !is.na(lon),
    lat >= -90 & lat <= 90,
    lon >= -180 & lon <= 180,
    !(lat == -1 & lon == -1)
  )


# Converter para sf
occs_sf <- st_as_sf(occs, coords = c("lon", "lat"), crs = 4326)

# Variáveis bioclimáticas WorldClim
bio_vars <- geodata::worldclim_global(var = "bio", res = 2.5, path = '/home/antonio')
bio_stack <- raster::stack(bio_vars)

# Limite político do Brasil
br <- ne_countries(scale = "large",
				   country = "Brazil",
				   returnclass = "sf")

# Converter variáveis ambientais para terra
bio_terra <- terra::rast(bio_stack)

# Reprojetar o limite do Brasil
br_proj <- st_transform(br, crs = crs(bio_terra))

# Recortar (crop + mask)
bio_crop <- terra::crop(bio_terra, vect(br_proj))
bio_mask <- terra::mask(bio_crop, vect(br_proj))

# Calcular zona UTM baseada na longitude média das ocorrências
mean_lon <- mean(occs$lon)
utm_zone <- floor((mean_lon + 180) / 6) + 1
utm_crs <- paste0("+proj=utm +zone=", utm_zone, " +datum=WGS84 +units=m +no_defs")

# Projetar para UTM
occs_utm <- st_transform(occs_sf, crs = utm_crs)
bio_utm <- terra::project(bio_mask, utm_crs)

# Converter para RasterStack
bio_utm_raster <- raster::stack()
for (i in 1:terra::nlyr(bio_utm)) {
  bio_utm_raster <- raster::addLayer(bio_utm_raster, raster::raster(bio_utm[[i]]))
}

# Pasta de saída
output_base_dir <- paste0('/home/antonio/mateus_results/maxent_vselection/')
dir.create(output_base_dir, showWarnings = FALSE)

# Lista de espécies
species_list <- unique(occs$species)


# Loop para cada espécie
for (sp in species_list) {
  message("Rodando para espécie: ", sp)
  
  sp_occurs <- occs_utm %>% 
	filter(species == sp) %>% 
	st_coordinates() %>% 
	as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  sdmdata_1sp <- setup_sdmdata(
	species_name = sp,
	occurrences = sp_occurs,
	predictors = bio_utm_raster,
	models_dir = output_base_dir,
	partition_type = "crossvalidation",
	cv_partitions = 5,
	cv_n = 1,
	n_back = 400,
	seed = 512,
	buffer_type = "mean",
	png_sdmdata = TRUE,
	select_variables = TRUE,
	clean_dupl = TRUE,
	clean_uni = FALSE,
	clean_nas = TRUE
  )
  message("Finalizado: ", sp)
}

for (sp in species_list) {
  message("Rodando para espécie: ", sp)
  
  sp_occurs <- occs_utm %>% 
	filter(species == sp) %>% 
	st_coordinates() %>% 
	as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  sp_maxnet <- do_many(
	species_name = sp,
	models_dir = output_base_dir,
	predictors = bio_utm_raster,
	png_partitions = TRUE,
	write_bin_cut = FALSE,
	equalize = TRUE,
	write_rda = FALSE,
	bioclim = TRUE,
	domain = FALSE,
	glm = TRUE,
	svmk = FALSE,
	svme = FALSE,
	maxent = TRUE,
	maxnet = FALSE,
	rf = FALSE,
	mahal = FALSE,
	brt = FALSE
  )
  message("Finalizado: ", sp)
}

for (sp in species_list) {
  message("Rodando para espécie: ", sp)
  
  sp_occurs <- occs_utm %>% 
	filter(species == sp) %>% 
	st_coordinates() %>% 
	as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  # Gerar modelo final (agregação das partições)
  final_model(
	species_name = sp,
	models_dir = output_base_dir,
	algorithms = c("maxent", "bioclim", "glm"),
	which_models = c("raw_mean"),
	consensus_level = 0.5,
	uncertainty = TRUE,
	overwrite = TRUE,
	write_rds = TRUE
  )
  message("Finalizado: ", sp)
}

for (sp in species_list) {
  message("Rodando para espécie: ", sp)
  
  sp_occurs <- occs_utm %>% 
	filter(species == sp) %>% 
	st_coordinates() %>% 
	as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  ens <- ensemble_model(species_name = sp,
					  occurrences = sp_occurs,
					  performance_metric = "pROC",
					  which_ensemble = c("average",
										 "best",
										 "frequency",
										 "weighted_average",
										 "median",
										 "pca",
										 "consensus"),
					  consensus_level = 0.5,
					  which_final = "raw_mean",
					  models_dir = output_base_dir,
					  overwrite = TRUE
  )                    
  message("Finalizado: ", sp)
}
message("Todos os modelos finalizados com ensemble.")


