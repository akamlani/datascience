# https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
Tab    <- read.csv("./ENB2012_data.csv", header=TRUE, sep=",", quote="\"", dec=".")
Tab    <- Tab[, 1:10]
Tab    <- Tab[rowSums(!is.na(Tab)) > 0, ]
X <- Tab[,1:8]
Y <- Tab[,9:10]
summary(Tab)
attach(Tab)

###### Explore Data/Visualization
# Y1 Most Correlated
metrics <- as.data.frame(cor(Tab, method="pearson"))
# Most Correlated for Y1: {X5, X1, X3, X7}
metrics_Y1 <- metrics[order(metrics$Y1, decreasing=TRUE),, drop=FALSE]
# Most Correlated for Y1: {X5, X1, X3, X7}
metrics_Y2 <- metrics[order(metrics$Y2, decreasing=TRUE),, drop=FALSE]
# scatterplot matrix
# TBD Highest Correlation Plots
pairs(Tab[, 1:8], main="Scatterplot Matrix")

##### Clusters (Are there any Clusters) via Methods
km <- kmeans(X, centers=10, iter.max=100, nstart=1, algorithm = "Hartigan-Wong", trace=FALSE)
plot(X, col=km$cluster,  main="KMeans Clustering")
points(km$centers, col = 1:ncol(X), pch = 8, cex = 2)
# clusters via optics (scaled entire column) 
# via optics, it looks like there are 3-5 clusters depending on the value of espilon
# with a smaller epsilon radius(0.05) ->k =5, with a larger epsilon(0.1) -> k=3
library(dbscan)
par(bg="white")
X_scaled = (X - min(X))/(max(X) - min(X))
res <- optics(X_scaled, eps =1, minPts = 50)
res <- optics_cut(res, eps = 0.05)
kNNdistplot(X_scaled, k=20)
plot(res,  main="Optics Clustering, eps=0.05, minPts=50")

##### Initial Feature Estimation
library(caret)
rfe_estimate <- function(X, Y)
{
  # verify some data with RFE directly from R
  # control <- rfeControl(functions=rfFuncs, method="repeatedcv", repeats=5)
  control <- rfeControl(functions=rfFuncs, method="cv", number=10)
  results <- rfe(X, Y, sizes=c(1:ncol(X)), rfeControl=control)
  print(results)
  predictors(results)
  plot(results, type=c("g", "o"),  main="RFE Estimate")
}
# Based on RFE and Clusters looks like there are 3-4 core groups of important variables
# Top Variables for Y1 (Elbow at 4 Variables): {X7, X8, X3, X2, X1}
rfe_estimate(X, Y[,1])
# Top Variables for Y2 (Elbow at 4 Variables): {X7, X8, X3, X1, X2}
rfe_estimate(X, Y[,2])

##### Outliers
# It doesn't really look like there are any outliers for this datset
# Although a few minimum observations could be dropped (e.g. value=0 in X7,X8)
for(i in 1:ncol(X))
{
  print(c(min(X[,i]), max(X[,i]), mean(X[,i]), median(X[,i]) ) )
}

##### Feature Selection (PCA for Exploration)
library(FactoMineR)
eval_pca <- function(X_in)
{
  # if scale.unit = FALSE -> a covariance matrix is used instead of a correlation matrix
  # currently selecting scale.unit = TRUE to normalize the data
  pca <- PCA(X_in, graph = TRUE, scale.unit = TRUE)
  print(pca$eig)
  print(pca$var$coord)
  head(pca$ind$coord)
  # look at the variance to evaluate maximized variance ~ 95% (2*sigma)
  plot(c(1:ncol(X_in)), pca$eig$`percentage of variance`,  main="PCA Evaluation")
  return (pca)
}
# It looks like the 95% (2* sigma is reached between 4-5 components), much like our RFE Estimate
pca_o <- eval_pca(X)
# check correlation (which features are mostly correlated with which component)
# note that unit scaling gives different results, so we choose to not unit scale to align with clusters
# {Component 1: {X5, X1}}, Component 2: {X3, X2, X5}, Component 3: {X8, X7}
summary(pca_o)
dimdesc(pca_o)
# Based on our results lets subset the data
labels <- c("X1", "X3", "X5", "X7")
pca_subset_o <- eval_pca(X[, labels])
# {Component 1: X5, X1}, {Component 2: X3, X5}
# This aligns with ~3 Components
summary(pca_subset_o)
dimdesc(pca_subset_o)


##### Feature Engineering: selected features
Tab_sel <- Tab[, c(labels, c("Y1", "Y2"))]
# matching labels
name_labels = c("RelativeCompactness", "SurfaceArea", "WallArea", "RoofArea", 
                "OverallHeight", "Orientation", "GlazingArea", "GlazingAreDist",
                "HeatingLoad", "CoolingLoad")

##### Train/Test Split (based on targete Parameter: Y1, Y2) - currently selected as Y1
# Split the data into Train/Test (based on Y1 Target Parameter)
target <- length(labels) + 1
partition       <- createDataPartition(y=Tab_sel[, target], p=0.75, list=FALSE)
training_part   <- Tab_sel[partition, ]
test_part       <- Tab_sel[-partition, ]
# Using the Train data, resplit into training part and validation part
partition_tr        <- createDataPartition(y=training_part[, target], p=0.75, list=FALSE)
training_part_sub   <- training_part[partition_tr, ]
validation_part_sub <- training_part[-partition_tr, ]

##### Predictions via Model Selection from 'target' parameter
# Choose the model to evaluate (Random Forest)
calc_error <- function(actual, predicted)
{
  res <- predicted - actual
  res_squared <- res^2
  mse <- mean(res_squared)
  rmse <- sqrt(mse)
  return (list(mse, rmse, res))
}

predict_model <- function(training_sub_in, validation_sub_in, testing_sub_in, metric_type, target_in)
{
  predictor_idx <- (target_in -1)
  model_base <- randomForest(training_sub_in[, target_in]~., 
                             data = training_sub_in[, 1:predictor_idx], 
                             importance=TRUE, ntree=200)
  
  metrics   <- as.data.frame(importance(model_base, metric_type))
  # plot the importance values
  varImpPlot(model_base, scale=TRUE)
  # predictions
  yhat_train <- predict(model_base, training_sub_in[, 1:predictor_idx]) 
  yhat_val   <- predict(model_base, validation_sub_in[, 1:predictor_idx]) 
  yhat_test  <- predict(model_base, testing_sub_in[, 1:predictor_idx])
  # calculate errors
  train_error <- calc_error(training_sub_in[, target_in], yhat_train)
  val_error   <- calc_error(validation_sub_in[, target_in], yhat_val)
  test_error  <- calc_error(test_part[, target_in], yhat_test)
  # plot the error as a function of trees
  plot(model_base)
  # log the RMSE errors
  cat("RMSE for Train/Validation/Test: ", 
      unlist(train_error[2]), unlist(val_error[2]), unlist(test_error[2]) )
  # TBD: Learning curves
  return (c(train_error, val_error, test_error))
}
model_errors <- predict_model(training_part_sub, validation_part_sub, test_part, metric_type=2, target)
# calculate cross-validation error
cv_base   <- rfcv(training_part_sub[, 1:(target-1)], training_part_sub[, target], cv.fold=5)
cv_k      <- as.data.frame(cv_base$error.cv[1])[,1]
cat("Cross-Validation Error: ", unlist(cv_k))

