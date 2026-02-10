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

max_batch = 1
max_fold = 1


models <- c("maxent","bioclim","glm","ensemble")

for (model in models){
	acc_mat <- matrix(NA, nrow = max_batch, ncol = max_fold + 1)
	f1s_mat <- matrix(NA, nrow = max_batch, ncol = max_fold + 1)

	for(BATCH in 1:max_batch){
		
		output_dir <- paste0('/home/antonio/mateus_results/maxent_vselection/val_results/')
		dir.create(output_dir, showWarnings = FALSE)
		
		df_geral <- NULL 
		for(FOLD in 1:max_fold){

			# Caminho principal dos modelos
			modelos_dir <- paste0('/home/antonio/mateus_results/maxent_vselection/')
			
			# CSV com coordenadas de teste
			test_csv <- paste0('/home/antonio/replicacao_mateus/dataset/validation_registros_species.csv')
			test_df <- read.csv(test_csv)

			# Renomear colunas de coordenadas
			test_df <- test_df %>%
			  dplyr::rename(
				lon = 'Registro..Longitude',
				lat = 'Registro..Latitude'
			  )
			  
			 
			test_df <- test_df %>%
				  filter(
					!is.na(lat),
					!is.na(lon),
					lat >= -90 & lat <= 90,
					lon >= -180 & lon <= 180,
					!(lat == -1 & lon == -1)
				  )


			# Converter para sf em WGS84
			test_sf <- st_as_sf(test_df, coords = c("lon", "lat"), crs = 4326)

			# Lista de espécies (pastas)
			species_dirs <- list.dirs(modelos_dir, recursive = FALSE)

			# Criar data.frame com coordenadas originais
			result_all <- test_df
			
			path = paste0(output_dir, model, '/')
			dir.create(path, showWarnings = FALSE)
			
			# Loop por espécie
			for (species_path in species_dirs) {
			  # Nome da espécie
			  species_name <- basename(species_path)
			  
			  if(model == "maxent"){
				  # Caminho do .tif
				  tif_file <- file.path(species_path, "present", "final_models", 
										paste0(species_name, "_maxent_raw_mean.tif"))
			  }
			  else if (model == "bioclim"){
				  # Caminho do .tif
				  tif_file <- file.path(species_path, "present", "final_models", 
										paste0(species_name, "_bioclim_raw_mean.tif"))
			  }
			  else if (model == "glm"){
				  # Caminho do .tif
				  tif_file <- file.path(species_path, "present", "final_models", 
										paste0(species_name, "_glm_raw_mean.tif"))
			  }
			  else if (model == "ensemble"){
				  # Caminho do .tif
				  tif_file <- file.path(species_path, "present", "ensemble", 
										paste0(species_name, "_ensemble_average.tif"))
			  }
			  else {
				  cat("MODELO INEXISTENTE")
			  }
			  
			  if (file.exists(tif_file)) {
				cat("Processando:", species_name, "\n")
				
				# Carrega raster
				model_rast <- rast(tif_file)
				
				# Reprojetar os pontos para o CRS do raster
				test_proj <- st_transform(test_sf, crs = crs(model_rast))
				test_vect <- terra::vect(test_proj)
				
				# Extrair valores
				predicted_values <- terra::extract(model_rast, test_vect)
				predicted_values <- predicted_values[, -1, drop = FALSE]  # Remove ID
				
				# Renomear coluna com nome da espécie
				colnames(predicted_values) <- species_name
				
				# Adicionar ao resultado geral
				result_all[[species_name]] <- predicted_values[[1]]
				
			  } else {
				cat("AVISO: Modelo não encontrado para:", species_name, "\n")
			  }
			}

			# Salvar resultado final
			# write.csv(result_all,  paste0(path,'maxent_test_fold', FOLD,'.csv'), row.names = FALSE)

			# Últimas 18 colunas: colunas de probabilidade por classe
			prob_cols <- tail(colnames(result_all), 34)
			class_names <- prob_cols  # nomes das espécies

			# Função para calcular top-k
			get_topk_flag <- function(probs_row, true_label, k) {
			  probs_named <- setNames(as.numeric(probs_row), class_names)
			  top_k <- names(sort(probs_named, decreasing = TRUE))[1:min(k, length(probs_named))]
			  return(as.integer(true_label %in% top_k))
			}

			# Calcula top-1, 3, 5, 10
			for (k in c(1, 3, 5, 10)) {
			  result_all[[paste0("top_", k, "_accuracy")]] <- mapply(
				FUN = function(true_label, probs_row) {
				  get_topk_flag(probs_row, true_label, k)
				},
				result_all$species,
				split(result_all[, prob_cols], seq(nrow(result_all)))
			  )
			}


			write.csv(result_all,  paste0(path, model, '_val_fold', FOLD,'.csv'), row.names = FALSE)
			
			df_geral <- rbind(df_geral, result_all)

			# Vetor com os nomes das colunas de acurácia
			col <- "top_1_accuracy"

			cat("\n Acurácias gerais médias:\n")

			
			mean_acc <- mean(result_all[[col]], na.rm = TRUE)
			cat(sprintf("%s: %.2f%%\n", col, mean_acc * 100))
			
			
			###########################################
			# F1 SCORE (MACRO)
			###########################################

			# Classe predita (top-1)
			predicted_class <- apply(
			  result_all[, prob_cols],
			  1,
			  function(x) class_names[which.max(x)]
			)

			# Classe verdadeira
			true_class <- result_all$species

			# Garantir conjunto comum de classes (como character)
			classes <- sort(unique(c(
			  as.character(true_class),
			  as.character(predicted_class)
			)))

			true_class <- factor(true_class, levels = classes)
			predicted_class <- factor(predicted_class, levels = classes)

			# Matriz de confusão
			conf_mat <- table(true_class, predicted_class)

			# Cálculo do F1 por classe
			f1_classes <- numeric(length(classes))
			names(f1_classes) <- classes

			for (i in seq_along(classes)) {
			  tp <- conf_mat[i, i]
			  fp <- sum(conf_mat[, i]) - tp
			  fn <- sum(conf_mat[i, ]) - tp
			  
			  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
			  recall    <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
			  
			  f1_classes[i] <- ifelse(
				precision + recall == 0,
				0,
				2 * precision * recall / (precision + recall)
			  )
			}

			# F1 macro
			f1_macro <- mean(f1_classes, na.rm = TRUE)

			cat(sprintf("\nF1-score macro: %.4f\n", f1_macro))
			
			acc_mat[BATCH, FOLD] <- mean_acc
			f1s_mat[BATCH, FOLD] <- f1_macro
		}
		write.csv(df_geral, paste0(path, model, '_val_global.csv'), row.names = FALSE)
		
		mean_acc <- mean(df_geral[[col]], na.rm = TRUE)
		acc_mat[BATCH, max_fold + 1] <- mean_acc
		cat(sprintf("\nGLOBAL %s: %.2f%%\n", col, mean_acc * 100))
		
		
		###########################################
		# F1 SCORE (MACRO)
		###########################################

		# Classe predita (top-1)
		predicted_class <- apply(
		  df_geral[, prob_cols],
		  1,
		  function(x) class_names[which.max(x)]
		)

		# Classe verdadeira
		true_class <- df_geral$species

		# Garantir conjunto comum de classes (como character)
		classes <- sort(unique(c(
		  as.character(true_class),
		  as.character(predicted_class)
		)))

		true_class <- factor(true_class, levels = classes)
		predicted_class <- factor(predicted_class, levels = classes)

		# Matriz de confusão
		conf_mat <- table(true_class, predicted_class)

		# Cálculo do F1 por classe
		f1_classes <- numeric(length(classes))
		names(f1_classes) <- classes

		for (i in seq_along(classes)) {
		  tp <- conf_mat[i, i]
		  fp <- sum(conf_mat[, i]) - tp
		  fn <- sum(conf_mat[i, ]) - tp
		  
		  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
		  recall    <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
		  
		  f1_classes[i] <- ifelse(
			precision + recall == 0,
			0,
			2 * precision * recall / (precision + recall)
		  )
		}

		# F1 macro
		f1_macro <- mean(f1_classes, na.rm = TRUE)
		
		f1s_mat[BATCH, max_fold + 1] <- f1_macro
		cat(sprintf("\n GLOBAL F1-score macro: %.4f\n", f1_macro))
		 
	}
	dir.create('/home/antonio/mateus_results/maxent_vselection/validation/', showWarnings = FALSE)
	write.csv(acc_mat, paste0('/home/antonio/mateus_results/maxent_vselection/validation/', model, '_acc_val_results.csv'), row.names = FALSE)
	write.csv(f1s_mat, paste0('/home/antonio/mateus_results/maxent_vselection/validation/', model, '_f1s_val_results.csv'), row.names = FALSE)
}


