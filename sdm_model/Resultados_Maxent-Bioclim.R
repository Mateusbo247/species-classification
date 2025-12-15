###########################################
#Rscript modelR_sdm.R -c <arquivo_de_treinamento.csv> -o <nome_do_arquivo_do_modelo> -lon <lon> -lat <lat> -m <nome_do_arquivo_do_modelo_treinado>
# Pacotes necessários para execução do Model-R
library(rJava)
library(raster)
library(modleR)
library(dplyr)
library(maps)
library(maptools)
library(rgeos)
## Creating an object with species names
getwd()
especies <- read.table("Animais/s_mico.csv", header=TRUE, sep= ",")
coord1sp <- especies

#Divisão entre conjunto de treinamento e teste.
#set <- sample(1:nrow(coord1sp), size = ceiling(0.7 * nrow(coord1sp)))
# Creating training data set (70% of species' records)
#train_set <- coord1sp[set,]
# Creating test data set (other 30%)
#test_set <- coord1sp[setdiff(1:nrow(coord1sp),set),]

test_folder <- "~/acm_article-pseudo"
################################################
# selecting only the first PCA axis
predictor <- example_vars
# transforming the data frame with the coordinates in a spatial object
pts <- SpatialPoints(coord1sp[,c(2,3)])
# ploting environmental layer
plot(predictor, legend = F)
#points(lat ~ lon, data=coord1sp)
especies[1,1]
occurrences = coord1sp[,-1]
occurrences
######################################################################
predictor

buf.env <- create_buffer(occurrences = coord1sp[,-1],
                         predictors = example_vars,
                         env_filter = TRUE,
                         env_distance = "centroid",
                         min_env_dist = 0.2
)

# using buf.env to generate 500 pseudoabsences
buf.env.p <- dismo::randomPoints(buf.env,
                                 n = 500,
                                 excludep = TRUE,
                                 p = pts)
# plotting environmental layer with background values
## environmental layer
plot(predictor[[1]],
     legend = FALSE, main = "environmental distance filter (centroid)")
## adding buff
plot(buf.env[[1]], add = TRUE, legend = FALSE, 
     col = scales::alpha("grey", 0.8), border = "black")
## adding buf.user.p
points(coord1sp[,c(2,3)], col = "red", pch = 10, cex=0.7)
points(buf.env.p, col = "blue", pch = 16)

###################################################

buf.env <- create_buffer(occurrences = coord1sp[,-1],
                         predictors = example_vars,
                         env_filter = TRUE,
                         env_distance = "mindist",
                         min_env_dist = 0.2
)

# using buf.env to generate 500 pseudoabsences
buf.env.p <- dismo::randomPoints(buf.env,
                                 n = 500,
                                 excludep = TRUE,
                                 p = pts)
# plotting environmental layer with background values
## environmental layer
plot(predictor[[1]],
     legend = FALSE, main = "environmental distance filter (mindist)")
## adding buff
plot(buf.env[[1]], add = TRUE, legend = FALSE, 
     col = scales::alpha("grey", 0.8), border = "black")
## adding buf.user.p
points(coord1sp[,c(2,3)], col = "red", pch = 10, cex = 0.7)
points(buf.env.p, col = "blue", pch = 16)

################################################################
#Divisão entre conjunto de treinamento e teste.
set <- sample(1:nrow(coord1sp), size = ceiling(0.7 * nrow(coord1sp)))
# Creating training data set (70% of species' records)
train_set <- coord1sp[set,]
# Creating test data set (other 30%)
test_set <- coord1sp[setdiff(1:nrow(coord1sp),set),]

###############################################################
m <- setup_sdmdata(species_name = especies[1,1],
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
                   boot_proportion = 0.7)
####################################################################
sp_maxnet <- do_any(species_name = especies[1,1],
                    algorithm = "maxnet",
                    predictors = example_vars,
                    models_dir = test_folder,
                    png_partitions = TRUE,
                    write_bin_cut = FALSE,
                    equalize = TRUE)
