library(randomForest)

predict_model <- function(training_sub_in, validation_sub_in, testing_sub_in, metric_type)
{
  # we could have used the exact same object from what we selected, but we don't store it
  # instead the training validated model will be on the selected feature space
  model_base <- randomForest(as.factor(Survived)~., 
                             data = training_sub_in, 
                             importance=TRUE, ntree=200)
  
  metrics   <- as.data.frame(importance(model_base, metric_type))
  yhat_base <- predict(model_base, validation_sub_in[, names(validation_sub_in) != "Survived"])
  cm_base   <- confusionMatrix(yhat_base, validation_sub_in[, "Survived"])
  accuracy  <- as.data.frame(t(cm_base$overall))$Accuracy
  # log the confusion matrix
  print("Results from Selected Features (Confusion, Accuracy): ")
  print(cm_base$table)
  print(accuracy)
  # plot the importance values
  varImpPlot(model_base, scale=TRUE)
  # evaluate the same fitted model on the test set to generalize (no 'Survived' values really exist)
  # Survived values are all 'null', this is empty column for convenience
  yhat_test <- predict(model_base, testing_sub_in[, names(testing_sub_in) != "Survived"])
  return (yhat_test)
}

rfe_estimate <- function(training_sub_in)
{
  # verify some data with RFE directly from R
  # control <- rfeControl(functions=rfFuncs, method="repeatedcv", repeats=5)
  control <- rfeControl(functions=rfFuncs, method="cv", number=10)
  results <- rfe(training_sub_in[, names(training_sub_in) != "Survived"], 
                 training_sub_in[,c("Survived")], sizes=c(1:ncol(training_sub_in)-1), rfeControl=control)
  print(results)
  predictors(results)
  plot(results, type=c("g", "o"))
}

train_model <- function(training_sub_in, validation_sub_in, metric_type)
{
  model_base <- randomForest(as.factor(Survived)~., 
                             data = training_sub_in, 
                             importance=TRUE, ntree=200)
  
  metrics   <- as.data.frame(importance(model_base, metric_type))
  yhat_base <- predict(model_base, validation_sub_in[, names(validation_sub_in) != "Survived"])
  cm_base   <- confusionMatrix(yhat_base, validation_sub_in[, "Survived"])
  accuracy  <- as.data.frame(t(cm_base$overall))$Accuracy
  cv_base   <- rfcv(training_sub_in[, names(validation_sub_in) != "Survived"], 
                    training_sub_in[, c("Survived")], cv.fold=5)
  cv_k      <- as.data.frame(cv_base$error.cv[1])[,1]
  return (list(accuracy, cv_k, metrics))
}

# Backward Feature Selection (BFS)
# Find least signifigant feature K in X
bfs <- function(training_in, validation_in, k, metric_type=2)
{
  f_space <- length(names(training_in)) -1
  train_space <- training_in
  validation_space <- validation_in
  lookup <- NULL
  
  if(k < 2) {k = 2}
  s <- length(f_space) - 1
  for (i in f_space:k) 
  {
    # evalulate cost function (improved accuracy)
    eval     <- train_model(train_space, validation_space, metric_type)
    imp      <- as.data.frame(eval[3])
    metric   <- imp[order(imp[,1], decreasing=TRUE),,drop=FALSE]
    lookup   <- rbind(lookup, c(paste(row.names(metric), collapse=', '), eval[1], eval[2]) )

    metrics_sub <-  metric[-nrow(metric), ,drop=FALSE]
    nlabels  <- row.names(metrics_sub)
    nlabels[[length(nlabels) +1]] <- "Survived"
    # update the train and validation functions based on the minimized features
    train_space <- train_space[, names(train_space) %in% nlabels]
    validation_space <- validation_space[, names(validation_space) %in% nlabels]
  }
  colnames(lookup) <- c("Features", "Accuracy", "CV Error")
  return (lookup)
}

# cluster models
library(fpc)
library(cluster)
library(dbscan)
cluster_analysis <- function(training_in)
{
  # Partitions: Check variables
  pm <- pamk(training_in, 20)
  cat("Num Clusters: ", pm$nc)
  plot(training_in, col=pm$pamobject$clustering)
  points(pm$pamobject$medoids, col = 1:ncol(training_in), pch = 8, cex = 2)
  # KMeans partitions: algorithms = c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen")
  km <- kmeans(training_in, centers=10, iter.max=100, nstart=1,
               algorithm = "Hartigan-Wong", trace=FALSE)
  plot(training_in, col=km$cluster)
  points(km$centers, col = 1:ncol(training_in), pch = 8, cex = 2)
  clusplot(training_in, km$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
  plotcluster(training_in, km$cluster)
  # Optics 
  par(bg="white")
  res <- optics(training_in, eps =1, minPts = 100)
  res <- optics_cut(res, eps = 0.1)
  kNNdistplot(training_in, k=20)
  plot(res)
}


# Initial Clustering Analysis based on a small feature selection
# A lot of overlap with 2 very distinct clusters
clabels <- c("Age", "Fare", "AgePclass") 
training_clust <- training_part_sub[,names(training_part_sub) %in% clabels]
cluster_analysis(training_clust)
clabels <- c("Age", "AgePclass", "FarePclass", "FarePclass2", "Fare")
training_clust  <- training_part_sub[,names(training_part_sub) %in% clabels]
cluster_analysis(training_clust)

# TBD: Remove highly correlated features via cor( method="pearson") and entropy
# TBD: RandomForest: hyperparameters (numtrees)
# TBD: gbm vs RandomForest

# note the search space includes our new features: FarePclass, FarePclass2
# acquire the 'k' search feature space based on feature importance and confusion matrix accuracy
search_sel <- as.data.frame(bfs(training_part_sub, validation_part_sub, k=2, metric_type=2))
lookup <- as.data.frame(lapply(search_sel, unlist))
feature_space <- lookup[order(lookup$Accuracy, decreasing = TRUE), ,drop=FALSE]
feature_sel   <- feature_space[1,]$Features
feature_sub   <- unlist(strsplit(toString(feature_sel), split=", "))
feature_acc   <- feature_space[1,]$Accuracy
# log the reesults to the console
# lower indexes are based on more variables (as it is a backwards search)
# It seems our the feature space mostly includes the new attributes: {FarePlass, FarePclass2}
print("Feature Space Results: ") 
print(feature_space)
# create feature space for selected model and perform evaluation and predictions on test set
feature_sub[length(feature_sub) +1] = "Survived"
feature_tr   <- training_part_sub[, names(training_part_sub) %in% feature_sub]
feature_val  <- validation_part_sub[, names(validation_part_sub) %in% feature_sub]
feature_test <- testing_sub[, names(testing_sub) %in% feature_sub]
yhat_eval    <- predict_model(feature_tr, feature_val, feature_test, metric_type=2)
# wrap up into format for kaggle
yhat_eval    <- as.data.frame(yhat_eval)
colnames(yhat_eval) <- "Survived"
yhat_eval$PassengerId <- row.names(yhat_eval)
yhat_eval <- yhat_eval[, c("PassengerId", "Survived")]
write.csv(yhat_eval, "./kaggle_titanic_submission.csv", row.names=FALSE)

# compare the results again RFE with smaller set of cross_validation
rfe_estimate(training_part_sub)


