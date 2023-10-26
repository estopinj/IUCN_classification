####### IMPORTS #######
library(usethis)
library(devtools)
library(reticulate)
library(IUCNN)
library(dplyr)
library(tidyr)
library(tidyverse)
library(magrittr)
library(caret)
library(dict)
library(nnet)
library(randomForest)
library(ggplot2)
use_condaenv("r-reticulate")
sklearn <- import("sklearn")

# Folder of species features
activ_folder <- "data/activations_dicts/"
# Log path
base_folder <- "logs"




####### PREDICTIVE FEATURES & LABELS #######

# IUCN Features
data("training_occ") #geographic occurrences of species with IUCN assessment
# All
iucnn_all <- iucnn_prepare_features(training_occ)   # Training features
# Geo
iucnn_geo <- iucnn_geography_features(x = training_occ)
# HF
iucnn_hf  <- iucnn_footprint_features(x = training_occ)




# choice=="species-occurences"
# Path of species features
activ_src <- paste0(activ_folder, "LonLat/")
# No weighting scheme
all_avg_activ     <- read.csv(paste0(activ_src,"all_avg_activations.csv"), header=TRUE, sep=";")
all_std_activ     <- read.csv(paste0(activ_src,"all_std_activations.csv"), header=TRUE, sep=";")
all_avg_std_activ <- read.csv(paste0(activ_src,"all_avg-std_activations.csv"), header=TRUE, sep=";")
all_sum_activ     <- read.csv(paste0(activ_src,"all_sum_activations.csv"), header=TRUE, sep=";")
all_3var_activ    <- read.csv(paste0(activ_src,"all_3var_activations.csv"), header=TRUE, sep=";")

# choice=="weighted_sum by relative proba of presence"
activ_src <- paste0(activ_folder, "activs_probas/all_occs/")
W_avg_activ     <- read.csv(paste0(activ_src,"AllPoints_DmCorr_avg_activations.csv"), header=TRUE, sep=";")
W_std_activ     <- read.csv(paste0(activ_src,"AllPoints_DmCorr_std_activations.csv"), header=TRUE, sep=";")
W_avg_std_activ <- read.csv(paste0(activ_src,"AllPoints_DmCorr_avg-std_activations.csv"), header=TRUE, sep=";")
W_sum_activ     <- read.csv(paste0(activ_src,"AllPoints_DmCorr_sum_activations.csv"), header=TRUE, sep=";")
W_3var_activ    <- read.csv(paste0(activ_src,"AllPoints_DmCorr_3var_activations.csv"), header=TRUE, sep=";")

# choice=="T + 10/rank"
activ_src <- paste0(activ_folder, "activs_probas/rank/")
rank_avg_activ     <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank10_avg_activations.csv"), header=TRUE, sep=";")
rank_std_activ     <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank10_std_activations.csv"), header=TRUE, sep=";")
rank_avg_std_activ <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank10_avg-std_activations.csv"), header=TRUE, sep=";")
rank_sum_activ     <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank10_sum_activations.csv"), header=TRUE, sep=";")
rank_3var_activ    <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank10_3var_activations.csv"), header=TRUE, sep=";")

# choice=="T + 1/rank"
activ_src <- paste0(activ_folder, "activs_probas/rank1/")
rank1_avg_activ     <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank1_avg_activations.csv"), header=TRUE, sep=";")
rank1_std_activ     <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank1_std_activations.csv"), header=TRUE, sep=";")
rank1_avg_std_activ <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank1_avg-std_activations.csv"), header=TRUE, sep=";")
rank1_sum_activ     <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank1_sum_activations.csv"), header=TRUE, sep=";")
rank1_3var_activ    <- read.csv(paste0(activ_src,"DmThresholdYRank_T8.75e-05_TOrank1_3var_activations.csv"), header=TRUE, sep=";")


# choice=="0dispersal"
activ_src <- paste0(activ_folder, "0dispersal/merged/")
zeroD_3var_activ <- read.csv(paste0(activ_src,"present_10rankThreshold0dispersal_TOrankT0disp_3var_activations.csv"), header=TRUE, sep=";")




# #Occurrences per species
# occs_count <- training_occ %>% group_by(species) %>% summarise(count = n())
# occs_count <- occs_count %>% arrange(desc(count))
occs_count <- iucnn_geo %>% select(species, tot_occ, uni_occ)



# List containing features & names
D_features <- list(#"iucnn_all"=iucnn_all,
                   "iucnn_geo"=iucnn_geo,
                   "iucnn_hf" =iucnn_hf,
                   "occs_count"=occs_count,
                   "all_avg_activ"=all_avg_activ,
                   "all_std_activ"=all_std_activ,
                   "all_avg_std_activ"=all_avg_std_activ,
                   "all_sum_activ"=all_sum_activ,
                   "all_3var_activ"=all_3var_activ,
                   
                   "W_avg_activ"=W_avg_activ,
                   "W_std_activ"=W_std_activ,
                   "W_avg_std_activ"=W_avg_std_activ,
                   "W_sum_activ"=W_sum_activ,
                   "W_3var_activ"=W_3var_activ,
                   
                   "rank_avg_activ"=rank_avg_activ,
                   "rank_std_activ"=rank_std_activ,
                   "rank_avg_std_activ"=rank_avg_std_activ,
                   "rank_sum_activ"=rank_sum_activ,
                   "rank_3var_activ"=rank_3var_activ,
                   
                   "rank1_avg_activ"=rank1_avg_activ,
                   "rank1_std_activ"=rank1_std_activ,
                   "rank1_avg_std_activ"=rank1_avg_std_activ,
                   "rank1_sum_activ"=rank1_sum_activ,
                   "rank1_3var_activ"=rank1_3var_activ,
                   
                   "zeroD_3var_activ"=zeroD_3var_activ
                   )

## LABELS ##
data("training_labels")

# Negate symbol
`%!in%` = Negate(`%in%`)

# labels from IUCN assessed species without weights associated:  3
unknwn_sp <- training_labels %>% filter(species %!in% weights32$species)

# # labels from IUCN assessed species not in the TEST set: 572!
# unknwn_sp_test <- training_labels %>% filter(species %!in% test_avg_activ$species)
# # labels from IUCN assessed species not in the TRAIN set: 17!
# unknwn_sp_train <- training_labels %>% filter(species %!in% train_avg_activ$species)

# labels from IUCN assessed species not in ALL OCCS: 17!
unknwn_sp_all <- training_labels %>% filter(species %!in% all_avg_activ$species)

# Common species support between activations & iucnn features: 872 --> 886
common_sp <- training_labels %>% filter(species %in% all_avg_activ$species)


####### FUNCTIONS #######
# Exp name & features log
log_features <- function(feat_list, sep=" "){
  L <- list()
  for (f in feat_list){L <- append(L, colnames(f)[-1])}
  return(paste(L, sep=sep))
}
exp_name     <- function(D, sep="_", plote=FALSE, data=FALSE){
  set.seed(seed=NULL)
  name <- paste0(D[["classifier"]], as.character(sample(1:1000, 1)))
  if (plote){
    name <- D[["classifier"]]
  }
  if (data){
    name <- paste(paste0(D[["features_names"]], collapse="\n"), name, sep=sep)
  }
  
  
  if (D[["classifier"]]=="MlpIucnn"){
    name <- paste(name,
                  gsub("_", ".", D[["hidden_layers"]]),
                  sep=sep)
    
  } else if (D[["classifier"]]=="RF"){
    name <- paste(name,
                  paste0("ntree",D[["ntree"]]),
                  paste0("mtry",D["mtry"]),
                  sep=sep)
    
  } else if (D[["classifier"]]=="SGD"){
    name <- paste(name,
                  paste(paste0("it" , D[["SGD_max_iter"]]), paste0("estop" , D[["SGD_early_stopping"]]), sep="-"),
                  paste(D[["grid_params"]][["loss"]], collapse="-"),
                  paste0("a", paste(D[["grid_params"]][["alpha"]]  , collapse="-")),
                  paste(D[["grid_params"]][["penalty"]], collapse="-"),
                  sep=sep)
    name <- gsub("modified_huber", "Mhub", name)
    name <- gsub("squared_hinge", "hing2", name)
    name <- gsub("perceptron", "perc", name)
    name <- gsub("FALSE", "Off", name)
    name <- gsub("TRUE", "On", name)
  }
  return(name)
}


# Macro-avg accuracy
computes_macro_avg <- function(labs, preds){
  df <- data.frame(labs, preds)
  macro.avg <- df %>%
    group_by(labs) %>%
    summarise(accuracy = sum(labs == preds) / n()) %>%
    summarise(macro.avg = mean(accuracy))
  return(macro.avg$macro.avg)
}

# Perfs
perfs <- function(labs, preds, D){
  L <- list()
  # Macro-avg
  macro.avg    <- computes_macro_avg(labs, preds)
  L[["macro"]] <- macro.avg
  # Factorize labels if need be
  if (D[["fact_labs"]]){
    labs   <- factor(labs, levels = D[["look.nums"]], labels = D[["look.labs"]])
  }
  if (D[["fact_preds"]]){
    if (D[["classifier"]]=="SGD"){
      preds <- factor(preds, levels = D[["look.labs"]], labels = D[["look.labs"]])
    } else{
      preds <- factor(preds, levels = D[["look.nums"]], labels = D[["look.labs"]])
    }
  }
  print(labs)
  print(preds)
  # Micro-avg and lot more
  u <- union(preds, labs)
  t <- table(factor(preds, u), factor(labs, u))
  c <- confusionMatrix(t)
  
  L[["micro"]] <- c$overall[['Accuracy']]
  if (D[["level"]]=="broad"){
    L[["sens"]] <- c$byClass[['Sensitivity']]
    L[["spec"]] <- c$byClass[['Specificity']]
  }
  else{ # detail level
    L[["sens"]] <- c$byClass[,'Sensitivity']
    L[["spec"]] <- c$byClass[,'Specificity']
  }
  # Verbose
  if (D[["verbose"]]){ print(c); print(paste("Macro-avg:", macro.avg))}
  return(L)
}

# Prepares labels
prepares_data_labels <- function(ffeatures, fold, D){
  
  # Test --
  test_slice  <- ffeatures %>% slice(fold)
  test        <- test_slice %>% select(-labels)
  test_labels <- test_slice %>% select(c(species,labels))
  
  # Test --
  data_slice  <- ffeatures %>% filter(species %!in% test_slice$species)
  data        <- data_slice %>% select(-labels)
  data_labels <- data_slice %>% select(c(species,labels))
  
  
  if (D[["classifier"]]=="MlpIucnn"){
    data_labels <- iucnn_prepare_labels(x = data_labels, y = data, level = D[["level"]])
    test_labels <- iucnn_prepare_labels(x = test_labels, y = test, level  = D[["level"]])
    
  }else if(D[["level"]]=="broad"){
    test_labels <- test_labels %>% mutate(labels = ifelse(labels == "VU" | labels == "EN" | labels == "CR", "Threat.", "Not T."))
    data_labels <- data_labels %>% mutate(labels = ifelse(labels == "VU" | labels == "EN" | labels == "CR", "Threat.", "Not T."))
  }
  
  if (D[["classifier"]]=="RF"  || D[["classifier"]]=="LogLinear" || D[["classifier"]]=="SGD"){
    # Merge of data & labels + factor labels
    data$species <- NULL
    test$species <- NULL
    data_labels$labels  <- factor(data_labels$labels, levels = D[["look.labs"]], labels = D[["look.labs"]])
    test_labels$labels  <- factor(test_labels$labels, levels = D[["look.labs"]], labels = D[["look.labs"]])
    # Type conversion
    data <- data.matrix(data)
    test <- data.matrix(test)
    # Scales data
    scaler <- sklearn$preprocessing$StandardScaler()
    scaler$fit(data)
    data <- scaler$transform(data)
    test <- scaler$transform(test)
    if (D[["classifier"]]=="RF" || D[["classifier"]]=="LogLinear"){
      # Type re-conversion
      data <- data.frame(data)
      test <- data.frame(test)
    }
    
  }
  
  return(list("data"=data, "data_labels"=data_labels, "test"=test, "test_labels"=test_labels))
}




# Prepares labels
prepares_data_labels_diffTest <- function(ffeatures, fold, D, ffeatures_test){
  
  # Test --
  test_slice  <- ffeatures_test %>% slice(fold)
  test        <- test_slice %>% select(-labels)
  test_labels <- test_slice %>% select(c(species,labels))
  
  # Test --
  data_slice  <- ffeatures %>% filter(species %!in% test_slice$species)
  data        <- data_slice %>% select(-labels)
  data_labels <- data_slice %>% select(c(species,labels))
  
  
  if (D[["classifier"]]=="MlpIucnn"){
    data_labels <- iucnn_prepare_labels(x = data_labels, y = data, level = D[["level"]])
    test_labels <- iucnn_prepare_labels(x = test_labels, y = test, level  = D[["level"]])
    
  }else if(D[["level"]]=="broad"){
    test_labels <- test_labels %>% mutate(labels = ifelse(labels == "VU" | labels == "EN" | labels == "CR", "Threat.", "Not T."))
    data_labels <- data_labels %>% mutate(labels = ifelse(labels == "VU" | labels == "EN" | labels == "CR", "Threat.", "Not T."))
  }
  
  if (D[["classifier"]]=="RF"  || D[["classifier"]]=="LogLinear" || D[["classifier"]]=="SGD"){
    # Merge of data & labels + factor labels
    data$species <- NULL
    test$species <- NULL
    data_labels$labels  <- factor(data_labels$labels, levels = D[["look.labs"]], labels = D[["look.labs"]])
    test_labels$labels  <- factor(test_labels$labels, levels = D[["look.labs"]], labels = D[["look.labs"]])
    # Type conversion
    data <- data.matrix(data)
    test <- data.matrix(test)
    # Scales data
    scaler <- sklearn$preprocessing$StandardScaler()
    scaler$fit(data)
    data <- scaler$transform(data)
    test <- scaler$transform(test)
    if (D[["classifier"]]=="RF" || D[["classifier"]]=="LogLinear"){
      # Type re-conversion
      data <- data.frame(data)
      test <- data.frame(test)
    }
    
  }
  
  return(list("data"=data, "data_labels"=data_labels, "test"=test, "test_labels"=test_labels))
}



# Classifier switch
classifier_switch <- function(data, data_labels, test, test_labels, D, names, i){
  
  switch(D[['classifier']], 
         MlpIucnn={
           # 5-FOLD CROSS-VALIDATION ---
           res_cv <- iucnn_train_model(x   = data,
                                       lab = data_labels,
                                       cv_fold = D[["cvfold"]],
                                       n_layers = D[["hidden_layers"]],
                                       seed = D[['seed']],
                                       balance_classes = FALSE,
                                       dropout_rate = D[["dropout_rate"]],
                                       path_to_output = "models/res_cv",
                                       overwrite = TRUE,
                                       verbose = D[["verbose"]])
           val.micro <- res_cv$validation_accuracy
           
           # PROD MODEL trained on train+val during avg_cv_best_epoch ---
           res_prod <- iucnn_train_model(x = data,
                                         lab = data_labels,
                                         production_model = res_cv,
                                         seed = D[['seed']],
                                         path_to_output = "models/res_prod",
                                         overwrite = TRUE,
                                         verbose = D[["verbose"]])
           # 10% TEST SET ACCURACY ---
           test_preds <- iucnn_predict_status(x = test, model = res_prod, return_IUCN = FALSE)$class_predictions
           test_perfs <- perfs(test_labels$labels$labels, test_preds, D)
         },
         
         RF={
           print('RF')
           # Train algorithm
           RFmodel <- randomForest(x     = data,
                                   y     = data_labels$labels,
                                   ntree = D[["ntree"]],
                                   mtry  = D[["mtry"]],
                                   na.action = na.omit)
           
           val.micro <- -1
           # Test set prediction
           test_preds <- predict(RFmodel, test)
           test_perfs <- perfs(test_labels$labels, test_preds, D)
         },
         
         LogLinear={
           # Fitting Multinomial log-linear regression to the Training set
           regressor <- multinom(formula = data_labels$labels ~ ., data = data)
           val.micro <- -1
           # Test set prediction
           test_preds <- predict(regressor, test)
           test_perfs <- perfs(test_labels$labels, test_preds, D)
         },
         
         SGD={
           print("SGD classifier")
           SGDClassif <- sklearn$linear_model$SGDClassifier(max_iter=D[["SGD_max_iter"]],
                                                            early_stopping=D[["SGD_early_stopping"]],
                                                            validation_fraction=1/D[["cvfold"]],
                                                            random_state=D[['seed']])
           # GRID SEARCH
           grid <- sklearn$model_selection$GridSearchCV(SGDClassif,
                                                        param_grid=D[["grid_params"]],
                                                        cv=D[["cvfold"]],
                                                        refit=TRUE)
           grid$fit(data, data_labels$labels)
           D[["bestGridSearch.params"]][[names[i]]]  <- grid$best_params_
           # Best cross-validation score
           val.micro <- grid$best_score_
           # Test
           test_preds <- grid$predict(test)
           test_perfs <- perfs(test_labels$labels, test_preds, D)
         },
         
         bar={
           # case 'bar' here...
           print('bar')    
         },
         
         {
           stop("Enter a valid classifier.") #default
         }
  )
  ## Val/Test performances integration
  D[["val.micro"]][[names[i]]]  <- val.micro
  D[["test.micro"]][[names[i]]] <- test_perfs[["micro"]]
  D[["test.macro"]][[names[i]]] <- test_perfs[["macro"]]
  D[["test.sens"]][[names[i]]]  <- test_perfs[["sens"]]
  D[["test.spec"]][[names[i]]]  <- test_perfs[["spec"]]
  D[["test.labels"]][[names[i]]]<- test_labels$labels
  D[["test.preds"]][[names[i]]] <- test_preds
  
  return(D)
}

# Avg perfs function
log_avg_perfs <- function(D){
  # Validation Micro accuracy
  D[["avg.val.micro"]] <- D[["val.micro"]] %>% unlist() %>% mean()
  
  # Test Micro & macro accuracy
  D[["avg.test.micro"]] <- D[["test.micro"]] %>% unlist() %>% mean()
  D[["avg.test.macro"]] <- D[["test.macro"]] %>% unlist() %>% mean()
  
  # Test Sensibility / Specificity
  if (D[["level"]]=="broad"){
    D[["avg.test.sens"]] <- D[["test.sens"]] %>% unlist() %>% mean()
    D[["avg.test.spec"]] <- D[["test.spec"]] %>% unlist() %>% mean()
  } else{ # detail
    D[["avg.test.sens"]] <- data.frame(D[["test.sens"]]) %>% apply(1,mean)
    D[["avg.test.spec"]] <- data.frame(D[["test.spec"]]) %>% apply(1,mean)
  }
  return(D)
}

# Log function
log <- function(ExpBatch, D){
  # log file name
  exp_log <- file.path(ExpBatch, paste(D[["EXP_NAME"]], Sys.time(), sep="_"))
  # Content
  log <- list()
  for (i in seq_along(D)){
    log <- append(log, c(names(D)[i], D[[i]], "\n"))
  }
  # Writing
  cat(unlist(log),file=paste0(exp_log,".txt"),sep="\n",append=TRUE)
  saveRDS(D, file=paste0(exp_log,".RData"))
}

# Recovers results before models comparison
recovers_batch_perfs <- function(ExpBatch, plote=TRUE){
  # Initializes results data structure
  res <- list()
  res$val.micro  <- list()
  res$test.micro <- list()
  res$test.macro <- list()
  
  # Recovers val.micro, test.micro, test.macro for exps in ExpBatch
  LExp <- Sys.glob(file.path(ExpBatch,'*.RData'))
  for (i in seq_along(LExp)) {
    Dexp <- readRDS(LExp[[i]])
    res$val.micro[[Dexp[["EXP_NAME"]]]]  <- unlist(Dexp$val.micro )
    res$test.micro[[Dexp[["EXP_NAME"]]]] <- unlist(Dexp$test.micro )
    res$test.macro[[Dexp[["EXP_NAME"]]]] <- unlist(Dexp$test.macro)
  }
  # res$val.micro  <- list(res$val.micro)
  # res$test.micro <- list(res$test.micro)
  # res$test.macro <- list(res$test.macro)
  return(res)
}

# For a given:
# *level in "detail"/"broad"
# *criterium "test.micro"/"test.macro"
# Recovers model with best avg performance for each data source and compare them
recovers_best_models <- function(base_folder, level, criterium, sep="--", plote=TRUE){
  # List of folders matching required level
  Lf <- Sys.glob(file.path(base_folder, paste(level,"*",sep="_")))
  # Initializes list of best model per folder
  Lb_Dexp <- list()
  
  # For loop on folders
  for (f in Lf) {
    cat("\t\t ***",f,"*** \n")
    # Recovers the list of experiments in this folder
    LExp <- Sys.glob(file.path(f,'*.RData'))
    
    if (length(LExp)==1){
      # Only 1 exp
      best_Dexp <- readRDS(LExp[[1]])
      best_avg  <- best_Dexp[[paste0("avg.",criterium)]]
      Lb_Dexp   <- append(Lb_Dexp, list(best_Dexp))
      cat("\t", best_Dexp[["EXP_NAME"]], "avg \t:", best_avg, "\n")
    } else if (length(LExp)>1){
      # 2 or more exps
      best_Dexp <- readRDS(LExp[[1]])
      best_avg <- best_Dexp[[paste0("avg.",criterium)]]
      for (exp in LExp){
        Dexp <- readRDS(exp)
        avg  <- Dexp[[paste0("avg.",criterium)]]
        cat("\t", Dexp[["EXP_NAME"]], "avg \t:", avg, "\n")
        if (avg > best_avg){
          best_Dexp <- Dexp
          best_avg  <- avg
        }
      }
      Lb_Dexp <- append(Lb_Dexp, list(best_Dexp))
    }
    cat("\t\t BEST:", best_Dexp[["EXP_NAME"]], "avg \t:", best_avg, "\n\n")
  }
  print(paste("Nb retained exps:", length(Lb_Dexp)))
  res <- list()
  for (Dexp in Lb_Dexp){
    name <- paste(paste0(Dexp[["features_names"]], collapse="\n"),
                  exp_name(Dexp, plote=TRUE, sep=sep), #Dexp[["EXP_NAME"]], # TODO: CHANGE here
                  sep=sep)
    res[[name]]  <- unlist(Dexp[[criterium]])
  }
  
  # ordering res by best avg
  L_avg      <- lapply(res, mean)
  sort_index <- order(as.double(L_avg), decreasing = TRUE) # CHANGED here
  print(sort_index)
  return(res[sort_index])
}


# Plots best models per data source to compare them
plots_best_results <- function(base_folder, lvl, crit){
  res  <- recovers_best_models(base_folder, lvl, crit, sep="--")
  
  if (lvl=="detail"){
    if (crit=="test.micro"){
      ylim <- c(0.22, 0.72)
    } else {
      ylim <- c(0.13, 0.53)
    }
  } else {
    ylim <- c(0.35, 0.90)
  }
  
  # Vertical box plot by group
  names(res) <- gsub("--", "\n", names(res))
  print(names(res))
  # names(res) <- c("IUCNN ref., RF*\nGeo. + HF features", "RF\nSDM avg/std/sum activations")
  # print(names(res))
  # names(res) <- gsub("-", "\n", names(res))
  boxplot(res, col = "white", ylim = ylim)
  # Points  c(3,2),
  stripchart(res, method = "jitter", pch = 19, col =  2:(length(res)+1),
             vertical = TRUE, add = TRUE)
  # Constructs averages
  means <- as.double(lapply(res, mean))
  names(means) <- names(res)
  points(means, col = "black", pch = 18, cex=1.75)
  
  title(paste("Crit :  ", crit,
              "\nLevel   :  ", lvl))
  par(mgp=c(3,-1.5,0))
  grid(nx = NA, ny = NULL)
}


####### PARAMETERS #######
# load("/home/jestopin/PHD/cactus/R_scripts/.RData")
D <- list()

## seed:
D[["seed"]] <- 1234L

## -- Features:      iucnn_all, iucnn_geo, iucnn_hf, occs_count, avg_activ, std_activ, avg_std_activ, sum_std_activ, 3var_activ
D[["features_names"]] <- c("all_3var_activ")
# D[["features_names_test"]] <- c("Tonly_avg_activ")
## --

# Auto
features_list   <- D_features[D[["features_names"]]]
# features_list_test <- D_features[D[["features_names_test"]]]
D[["features"]] <- log_features(features_list)

## Level:         "broad" or "detail"
D["level"] <- "detail"


# test/cv folds:
D["kfold"]  <- 10
D["cvfold"] <- 5L


# TODO: ADD SVM / sgd-classifier
## Classifier:    "MlpIucnn", "RF", "LogLinear", "SGD"
D["classifier"] <- "RF"

## Classifier parameters:
D["verbose"]       <- TRUE
D["hidden_layers"] <- "9_9_9"       # MLP hidden_layers: "N_M_L" format
D["dropout_rate"]  <- 0.1
D["ntree"] <- 10000
D["mtry"]  <- 100 #floor(sqrt(ncol(data))) # 100

D[["SGD_early_stopping"]] <- TRUE
D[["SGD_max_iter"]] <- 1000L
# SGD classif CV GRID SEARCH
params <- list()
params[["loss"]]    <- list("hinge", "modified_huber", "perceptron") #  "log", "squared_hinge", 
params[["alpha"]]   <- list(0.0001, 0.001, 0.01, 0.1)
params[["penalty"]] <- list("l2", "l1") # , "elasticnet"
D[["grid_params"]]  <- params

## NAME
D[["EXP_NAME"]] <- exp_name(D)

# --- --- --- ---
 

####### VARIABLES #######

# ExpBatch folder
ExpLevelData <- paste(D[["level"]], paste(D[["features_names"]], collapse="-"), sep="_")
ExpBatch     <- file.path(base_folder, ExpLevelData)
dir.create(ExpBatch, showWarnings = FALSE)
setwd(ExpBatch)

# Train** Final features
ffeatures <- features_list %>% reduce(inner_join, by='species')
# + labels
# ffeatures <- merge(ffeatures, training_labels, by = 'species') %>% drop_na()
ffeatures <- merge(ffeatures, common_sp, by = 'species') %>% drop_na()
# labels <- training_labels %>% filter(species %in% ffeatures$species)
# labels <- labels[match(ffeatures$species, labels$species),]

# Test** Final features
# ffeatures_test <- features_list_test %>% reduce(inner_join, by='species')
# + labels
# ffeatures <- merge(ffeatures, training_labels, by = 'species') %>% drop_na()
# ffeatures_test <- merge(ffeatures_test, common_sp, by = 'species') %>% drop_na()
# labels <- training_labels %>% filter(species %in% ffeatures$species)
# labels <- labels[match(ffeatures$species, labels$species),]



# Lookup table
if (D[["level"]]=="broad"){
  D[["look.nums"]] <- c(0, 1)
  D[["look.labs"]] <- c("Not T.", "Threat.")
} else{
  D[["look.nums"]] <- c(0, 1, 2, 3, 4)
  D[["look.labs"]] <- c("LC", "NT", "VU", "EN", "CR")
}

# factorize test_preds/labels before computing perfs?
if (D[["classifier"]]=="MlpIucnn" ){
  D[["fact_labs"]]  <- TRUE
  D[["fact_preds"]] <- TRUE
} else if (D[["classifier"]]=="SGD" ){
  D[["fact_labs"]]  <- FALSE
  D[["fact_preds"]] <- TRUE
} else{
  D[["fact_labs"]]  <- FALSE
  D[["fact_preds"]] <- FALSE
}


# Dictionary initialization
D[["val.micro"]]  <- list()
D[["test.micro"]] <- list()
D[["test.macro"]] <- list()
D[["test.sens"]]  <- list()
D[["test.spec"]]  <- list()


####### 10-FOLD TEST #######

# Sets seed for reproducibility
set.seed(D[["seed"]])
folds <- createFolds(1:nrow(ffeatures), k = D[["kfold"]])

for (i in seq_along(folds)) {
  print(names(folds)[i])
  
  # Prepares data & labels
  split       <- prepares_data_labels(ffeatures, folds[[i]], D)
  # split       <- prepares_data_labels_diffTest(ffeatures, folds[[i]], D, ffeatures_test)
  data        <- split[["data"]]
  data_labels <- split[["data_labels"]]
  test        <- split[["test"]]
  test_labels <- split[["test_labels"]]
  
  ### Classifier switch ###
  D <- classifier_switch(data, data_labels, test, test_labels, D, names(folds), i)
  
}

### AVG PERFS ###
D <- log_avg_perfs(D)

### LOG ###
print(D)
log(ExpBatch, D)



####### MODEL COMPARISON ----
if (D[["level"]]=="detail"){
  hist_labels <- factor(ffeatures$labels , levels=c("LC", "NT", "VU", "EN", "CR"))
  barplot(table(hist_labels), ylab = "Frequency", main = "IUCN status distribution", col=c("ghostwhite", "grey", "yellow", "orange", "red"))
} else{
  binary_labels <-  ffeatures %>% mutate(labels = ifelse(labels == "VU" | labels == "EN" | labels == "CR", "Threat.", "Not T.")) %>% select(labels)
  ggplot(binary_labels, aes(labels)) +
    geom_bar(fill = c("#0073C2FF",'red')) +
    ylim(-5,475) +
    geom_text(stat='count', aes(label=..count..), vjust=-1) +
    ggtitle("IUCN binary status distribution") +
    theme(plot.title = element_text(hjust = 0.5))
}


res <- recovers_batch_perfs(ExpBatch, plote=FALSE)

## PLOT
var <- "test.micro"


# Vertical box plot by group
names(res[[var]]) <- gsub("_", "\n", names(res[[var]]))
boxplot(res[[var]], col = "white", ylim = c(0.60, 0.90)) #c(0.22, 0.72))
# Points
stripchart(res[[var]], method = "jitter", pch = 19, col = 1:length(res[[var]]),
           vertical = TRUE, add = TRUE)
title(paste("Variable :  ", var,
            "\nLevel :  ", D[["level"]],
            "\nFeatures :  ", paste(D[["features_names"]], collapse = " / ")))
par(mgp=c(3,-1.5,0))
grid(nx = NA, ny = NULL)


####### DATA SOURCES COMPARISON ----

lvl  <- "detail"         # "detail / "broad"
crit <- "test.micro"     # "test.micro" / "test.macro"

# selected_folder <- base_folder
selected_folder <- paste0(base_folder,"/Selection_DETAIL")

plots_best_results(selected_folder, lvl, crit)


################################################################################
# LINEAR MODEL ----

# Fitting Multinomial log-linear regression to the Training set
regressor <- multinom(formula = data_labels$labels ~ ., data = data)

# Test set prediction
test_preds <- predict(regressor, test)
# Test set performance
xtab      <- table(test_preds, test_labels$labels)
confusionMatrix(xtab)



# SGDClassifier ----

# Scales data
scaler <- sklearn$preprocessing$StandardScaler()
scaler$fit(data)
data   <- scaler$transform(data)
test   <- scaler$transform(test)


# SGD Classifier
SGDClassif <- sklearn$linear_model$SGDClassifier(max_iter=D[["SGD_max_iter"]],
                                                 early_stopping=D[["SGD_early_stopping"]],
                                                 validation_fraction=1/D[["cvfold"]])
# GRID SEARCH
grid <- sklearn$model_selection$GridSearchCV(SGDClassif,
                                             param_grid=D[["grid_params"]],
                                             cv=D[["cvfold"]])
grid$fit(data, data_labels$labels)
print(grid$best_params_)
# Best cross-validation score
micro.avg <- grid$best_score_
# Test
test_preds <- grid$best_estimator_$predict(test)
perfs(test_labels$labels, test_preds, D)





# RANDOM FOREST ----


# ************
lvl = "detail"
# ************

# Prepares RF & IUCNN features & labels
# RF 
RF_choice   <- c("all_avg_std_activ")   # all_avg_std_activ
RF_features <- D_features[RF_choice] %>% reduce(inner_join, by='species')
RF_features <- merge(RF_features, common_sp, by = 'species') %>% drop_na()

RF_labels   <- RF_features %>% select(c(species,labels))
if(lvl=="broad"){
  RF_labels <- RF_labels %>% mutate(labels = ifelse(labels == "VU" | labels == "EN" | labels == "CR", "Threat.", "Not T."))
  RF_labels$labels  <- factor(RF_labels$labels, levels = c("Not T.", "Threat."), labels = c(0, 1))
} else{
  RF_labels$labels  <- factor(RF_labels$labels, levels = c("LC", "NT", "VU", "EN", "CR"))
}

RF_features$species <- NULL
RF_features$labels <- NULL

# Type conversion
RF_features <- data.matrix(RF_features)
# Scales data
scaler <- sklearn$preprocessing$StandardScaler()
scaler$fit(RF_features)
RF_features <- scaler$transform(RF_features)
# Type re-conversion
RF_features <- data.frame(RF_features)

print('RF')
# Train algorithm
RFmodel <- randomForest(x     = RF_features,
                        y     = RF_labels$labels,
                        ntree = D[["ntree"]],
                        mtry  = D[["mtry"]],
                        na.action = na.omit)

# ALL SPECIES PREDICTION
test  <- D_features[RF_choice] %>% reduce(inner_join, by='species')

geo_all_occs       <- iucnn_geography_features(all_occs)

if (lvl=="detail"){
  hf_all_occs         <- iucnn_footprint_features(x = all_occs)
  all_geo_hf_features <- list(geo_all_occs, hf_all_occs) %>% reduce(inner_join, by='species')
  all_geo_hf_species <- data.frame(species=all_geo_hf_features$species)
  test <- merge(test, all_geo_hf_species, by = 'species') %>% drop_na()
} 

RF_species <- data.frame(species=test$species)
test$species <- NULL

# Conversions
# Type conversion
test <- data.matrix(test)
# Scales data
test <- scaler$transform(test)
# Type re-conversion
test <- data.frame(test)


# # Test set prediction
preds_RF <- predict(RFmodel, test)
# test_perfs <- perfs(test_labels$labels$labels, test_preds, D)
# 
# test_probs <- predict(RFmodel, test, type = "prob")
# test_votes <- predict(RFmodel, test, type = "vote", norm.votes = FALSE)

if (lvl=="broad"){
  preds_RF <- as.numeric(preds_RF) -1
  preds_RF <- factor(preds_RF, levels = c(0, 1), labels = c("Not Threatened", "Threatened"))
}

print(preds_RF)
plot(preds_RF, col = c("white", "gray", "yellow", "orange", "red"))



# IUCNN Model Testing ----

# Labels preparation
if (lvl=="broad"){
  features <- iucnn_geo
} else{
  geo_hf_features <- list(iucnn_geo, iucnn_hf) %>% reduce(inner_join, by='species')
  features        <- geo_hf_features
  features_species<- data.frame(species = features$species)
  # Restrains species to same as RF
  training_labels <- merge(training_labels, features_species, by = 'species') %>% drop_na()
}


labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features,
                                     level = lvl) # Training labels
# Model testing
mod_test <- iucnn_modeltest(x = features,
                            lab = labels_train,
                            mode = "nn-class",
                            dropout_rate = c(0.0, 0.1), # Best 0.1
                            n_layers = c("9_9_9"),
                            cv_fold = 5,
                            model_outpath = "france_occs10",
                            logfile = "france_occs_logfile10.txt")

# Select best model
m_best <- iucnn_best_model(x = mod_test,
                           criterion = "val_acc",
                           require_dropout = TRUE)

# Inspect model structure and performance
summary(m_best)
plot(m_best)

# Train the best model on all training data for prediction
m_prod <- iucnn_train_model(x = features,
                            lab = labels_train,
                            production_model = m_best,
                            overwrite = TRUE)


# IUCNN Predictions ----
data("prediction_occ") # No ground truth
features_predict <- iucnn_prepare_features(prediction_occ)
geo_predict      <- iucnn_geography_features(prediction_occ) # Prediction features

# All occurrences
ref                <- "/home/jestopin/PHD/data/occurrences/orchids_extraction_ref/DeepOrchidSeries.csv"
df_ref             <- read.csv(file = ref, sep = ";")
all_occs           <- df_ref %>% select(canonical_name, decimallongitude, decimallatitude)
names(all_occs)[1] <- "species"

# geo_all_occs       <- iucnn_geography_features(all_occs)
if (lvl=="broad"){
  # Restrains species to same as RF
  geo_all_occs       <- merge(geo_all_occs, RF_species, by = 'species') %>% drop_na()
} else {
  # hf_all_occs         <- iucnn_footprint_features(x = all_occs)
  all_geo_hf_features <- list(geo_all_occs, hf_all_occs) %>% reduce(inner_join, by='species')
  features            <- all_geo_hf_features
  # Restrains species to same as RF
  features <- merge(features, RF_species, by = 'species') %>% drop_na()
}



# France occurrences
fr_occs           <- df_ref %>% filter(bot_code == "FRA")
fr_occs           <- fr_occs %>% select(canonical_name, decimallongitude, decimallatitude)
names(fr_occs)[1] <- "species"
geo_fr_occs       <- iucnn_geography_features(fr_occs)
hf_fr_occs        <- iucnn_footprint_features(fr_occs)

fr_features       <- list(geo_fr_occs, hf_fr_occs) %>% reduce(inner_join, by='species')




# Predict RL categories for target species
preds_iucnn <- iucnn_predict_status(x = features, model = m_prod) # , return_raw = TRUE)
preds_iucnn <- preds_iucnn$class_predictions

if (lvl=="broad"){
  preds_iucnn <- factor(preds_iucnn, levels = c("Not Threatened", "Threatened"), labels = c("Not Threatened", "Threatened"))
}else {
  preds_iucnn <- factor(preds_iucnn, levels = c("LC", "NT", "VU", "EN", "CR"))
}

plot(preds_iucnn, col = c("white", "gray", "yellow", "orange", "red"))


# Plots BOTH ----

df_RF       <- data.frame(table(preds_RF))
df_RF$Model <- "RF with SDM avg/std activations"
df_RF       <- rename(df_RF, "Status" = "preds_RF", "Count" = "Freq")

df_iucnn       <- data.frame(table(preds_iucnn))
df_iucnn       <- rename(df_iucnn, "Status" = "preds_iucnn", "Count" = "Freq")

if (lvl=="broad"){
  df_iucnn$Model <- "MLP with iucnn geo. features"
} else{
  df_iucnn$Model <- "MLP with iucnn geo. & HF features"
}

df <- rbind(df_RF, df_iucnn)

ggplot(df,
       aes(x = Status,
           y = Count,
           fill = Model)
       ) +
  geom_bar(stat = "identity",
           position = "dodge")


table <- table(preds_RF, preds_iucnn)
c <- confusionMatrix(table)
mosaicplot(table)





# Export to csv ----

# write.csv(training_labels,
#           "/home/jestopin/PHD/cactus/R_scripts/exports/IUCN_GroundTruth.csv",
#           row.names = TRUE)
# 
# 
# pred_labels_all <- data.frame(species = pred_all$names,
#                          labels = pred_all$class_predictions)
# 
# write.csv(pred_labels_all,
#           "/home/jestopin/PHD/cactus/R_scripts/exports/pred_labels_all.csv",
#           row.names = TRUE)


# pred_labels_fr <- data.frame(species = pred_all$names,
#                          labels = pred_all$class_predictions)
# write.csv(pred_labels_fr,
#           "/home/jestopin/PHD/cactus/R_scripts/exports/pred_labels_fr.csv",
#           row.names = TRUE)

# Ensembling test ----

e1 <- test_probs
e2 <- pred_all$mc_dropout_probs %>%
  head(nrow(e1))

X <- list(e1, e2)
Y <- do.call(cbind, X)
Y <- array(Y, dim=c(dim(X[[1]]), length(X)))
m <- apply(Y, c(1, 2), mean, na.rm = TRUE)

m_idx   <- max.col(m) -1
preds_m <- factor(m_idx, levels = D[["look.nums"]], labels = D[["look.labs"]])

