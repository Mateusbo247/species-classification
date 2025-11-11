###########################################
#Rscript modelR_sdm.R -c <arquivo_de_treinamento.csv> -o <nome_do_diretorio_do_modelo> -lon <lon> -lat <lat> -m <nome_do_diretorio_do_modelo_treinado>
#
###########################################
#Instalação do Model-R.

install_user_dir <- "~/R"
.libPaths( c( .libPaths(), install_user_dir ) )

## Relação de Pacotes a serem instalados.
packages <- c("rJava", "raster", "dplyr", "devtools", "maps", "maptools", "rgeos", "argparse", "remotes")
instpack <- packages[!packages %in% installed.packages()]
if (length(instpack) > 0) {
  install.packages(packages[!packages %in% installed.packages()], lib=install_user_dir)
}

pack <- c("modleR")
instp <- pack[!pack %in% installed.packages()]
if (length(instp) > 0) {
  remotes::install_github("Model-R/modleR", build = TRUE, lib=install_user_dir)
}

# Pacotes necessários para execução do Model-R
library(rJava)
library(raster)
library(modleR)
library(dplyr)
library(maps)
library(maptools)
library(rgeos)
library(argparse)
#################################################################################

# create parser object
parser <- ArgumentParser()

# specify our desired options 
# by default ArgumentParser will add an help option 

#---------------------------------------------------
parser$add_argument("-c", "--csv", default=FALSE,
                    help="CSV file of the form 'species,longitude,latitude[,date]' in which the general model will be built upon. If not given, it is expected that the collaborator-specific CSV file will be given in order to (solely) built the model.")
#---------------------------------------------------
parser$add_argument("-o", "--outputmodel", default=FALSE, help="Name of model file")
#---------------------------------------------------
parser$add_argument("-lon", "--lon", default=FALSE, type=numeric, 
                    help="Longitude of point of interest, i.e., the longitude of the point whose species prediction is to be computed.")
#---------------------------------------------------
parser$add_argument("-lat", "--lat", default=FALSE, type=numeric, 
                    help="Latitude of point of interest, i.e., the latitude of the point whose species prediction is to be computed.")
#---------------------------------------------------
parser$add_argument("-m", "--model", default=FALSE, help="Name of train model. ")
#---------------------------------------------------
# get command line options, if help option encountered print help and exit.
args <- parser$parse_args()


#######################################################################################
## Geração do modelo de distribuição de Espécies.
if (args$csv){
  
  if (args$modelfile){
  
    ## Creating an object with species names
    training_set <- read.table(args$csv, header=TRUE, sep= ",")
    species <- unique(training_set$species) 
    
    ##########################################################################
    # Geração dos Modelos de todas as espécies
    for (i in 1:length(species)) {
      sp <- species[i]
      occs <- training_set[training_set$species == species, c("lon", "lat")] 
      pts1 <- occs
     
      ##Criação do Buffer Mediana para o parâmetro min_geog_dist.
      names(pts1) = c("lon", "lat")
      buffer_type = "median"
      if(buffer_type=="median"){
            dist.buf <- median(spDists(x = pts1, longlat = FALSE, segments = TRUE))
      }
      
      ##Configuração das Pseudo-Ausências.
      setup_sdmdata(species_name = sp,
                    occurrences = occs,
                    models_dir = args$outputmodel,
                    predictors = example_vars,
                    buffer_type = "median",
                    min_geog_dist = 0.3*dist.buf
                    write_buffer = TRUE,
                    clean_dupl = FALSE,
                    clean_nas = FALSE,
                    clean_uni = FALSE,
                    png_sdmdata = FALSE,
                    n_back = length(occs),
                    partition_type = "crossvalidation",
                    cv_partitions = 1,
                    cv_n = 1
      )
    }
  
 ##Modelos são gerados por meio do algoritmo MaxEnt. 
    for (i in 1:length(species)) { 
      sp <- species[i]
      write.csv(species, "./args$outputmodel/species.csv", row.names = FALSE)
      do_any(species_name = sp,
             algorithm = "maxnet",
             predictors = example_vars,
             models_dir = args$outputmodel,
             png_partitions = FALSE,
             write_bin_cut = FALSE,
             equalize = TRUE)
    }
   
  }
}

##Teste do modelo gerado para verificação da espécie de um registro adicionado.
if (args$model){
  if (args$lon & args$lat){
    
    species <- read.table("./args$model/species.csv", header=TRUE, sep= ",") ##Leitura do arquivo contendo a lista de espécies.
    
    ##Modelo Final para verificação da classe de um novo registro.
    for (i in 1:length(species)) {
      sp <- species[i]
      occ <- [args$lon, args$lat]
      sp_final <- final_model(species_name = sp, 
                              occurrences = occ,
                              algorithms = "maxnet",
                              models_dir = args$model,
                              which_models = c("raw_mean"),
                              overwrite = TRUE)
    
      names(score) <- colnames(accuracy) ##Criando tabela para armazenar resultados da predição.
      score[i] <- sp_final.accuracy[,i]
    }


# Normalize the scores (i.e., calculate the probabilities, sum=1)
   soma <- sum(score)
   
   for (i in 1:length(score)){
      score[i] = score[i]/soma
    }

## Ordenação dos resultados para apresentação.
    res1 <- sort(score[,2], decreasing = T)
    ##Vetor com os índices após a ordenação da probabilidade.
    res2 <- order(score[,2], decreasing = T)
    
    
    print(sprintf("\n Probabilities \n"))
    sprintf("%s: %f", score[res2,1], res1) ##Retorna a espécie que possui índice adequado e sua probabilidade.
    print(sprintf("\n Predicted species: \n"))
    sprintf("%s (%.2f\%)", score(res2[1],1), 100*res1[1]) ##Retorna a espécie mais provável e sua probabilidade.


# TODO: Normalizar o vetor sp_final (soma igual a 1) e ordená-lo de forma decrescente (sem perder a espécie correspondente)

# TODO: Colocar a saída nesse formato (exemplo):
#Probabilities {'Macaco': 0.7322707911563455, 'Ave aquática ou marinha': 0.029326110818689317, 'Outro': 0.021900286516350394, 'Tartaruga': 0.013440452778876308, 'Anfíbio: Perereca/Rã/Salamandra/Sapo': 0.011490965661972148, 'Mico': 0.011417966949517716, 'Pássaro': 0.011152861429214232, 'Ave - Outras (Jacu/Siriema/Tucano)': 0.01057412269633423, 'Não identificado': 0.01028833575440146, 'Ave de rapina: Águia/Coruja/Gavião': 0.009866875764015272, 'Jacaré': 0.009121015428196905, 'Morcego': 0.00850513900137147, 'Cachorro do Mato/Raposa/Graxaim': 0.008377986429993314, 'Tatu': 0.008300151164218837, 'Gambá/Cuíca': 0.007965010216618265, 'Cobra': 0.007919609093504821, 'Preguiça': 0.007563312772891945, 'Capivara': 0.00741613104648657, 'Lobo Guará': 0.0072065389337464284, 'Cotia/Mocó/Preá': 0.006968668975240073, 'Quati/Guaxinim/Furão': 0.006893284115224282, 'Gato do Mato/Jaguatirica/Jaguarundi': 0.006860722465055296, 'Jabuti': 0.006563015981977496, 'Rato/Camundongo': 0.006445726557905018, 'Lagarto': 0.006443133767377127, 'Arara/Papagaio/Periquito': 0.006298688460538454, 'Tamanduá': 0.005865731719396264, 'Paca': 0.0055733515017832745, 'Onça Parda (Suçuarana/Leão Baio/Onça Vermelha)': 0.004275113892883759, 'Anta': 0.003708898949873832}
#
#Predicted species:
#Macaco (73.23%)

  }
}
