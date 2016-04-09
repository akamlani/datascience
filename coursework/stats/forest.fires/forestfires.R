# http://archive.ics.uci.edu/ml/datasets/Forest+Fires
# Forest Fires
Tab    <- read.csv("./forestfires.csv", header=TRUE, sep=",", quote="\"", dec=".")
Tab_Num <- Tab[, names(Tab) != "month" & names(Tab) != "day"]
X <- Tab[, 1:12]
Y <- Tab[, 13]
##### Explore the data
# There is not much orrelation between numerical data and predicted area
attach(Tab_Num)
cor(Tab_Num)
# lets perform a binning of Y and a histogram
sd(Y)
plot(1:length(Y),Y)
plot(1:length(Y[Y >25]), Y[Y>25])
# lets review the samples of the output area (Y)
breaks <- c(-1, 1,10,25,200,800)
bins<-cut( Y, breaks)
hist(Y, bins)

# scatterplot matrix for select features
# aggregation for burnt areas in time(day, month) and weather (rain, RH, wind, temperature) conditions
pairs(area~temp+wind+rain+RH+day+month, data=Tab)
# determine if there are any outliers reltive to the area(Y)
outliers <- Tab[Tab$area > 600,]
outlier_idx <- c(as.numeric( rownames(outliers) ) )
Tab_filtered <- Tab[-outlier_idx, ]
pairs(area~temp+wind+rain+RH+day+month, data=Tab_filtered)

##### Explore via PCA: Feature Selection
library(FactoMineR)
eval_pca <- function(X_in, cat_col)
{
  # if scale.unit = FALSE -> a covariance matrix is used instead of a correlation matrix
  # currently selecting scale.unit = TRUE to normalize the data
  pca <- PCA(X_in, graph = TRUE, scale.unit = TRUE, quali.sup=cat_col)
  print(pca$eig)
  print(pca$var$coord)
  head(pca$ind$coord)
  
  # look at the variance to evaluate maximized variance ~ 95% (2*sigma)
  plot(c(1:length(pca$eig$`percentage of variance`)), pca$eig$`percentage of variance`,  main="PCA Evaluation")
  return (pca)
}
X <- Tab_filtered[, 1:12]
Y <- Tab_filtered[, 13]
# If not scaled to unit: 2 components: {DMC, DC}
# Else Eight Components: e.g. {X/Y, RH, wind, rain, temp...}
pca_o <- eval_pca(X, 3:4)
summary(pca_o)
dimdesc(pca_o)

##### Spatial Clusters (LDA, kNN)
##### Feature Engineering (derived BUI or FWI Variable)
##### Display Points on map

# RFE Estimate
library(caret)
rfe_estimate <- function(X, Y)
{
  # verify some data with RFE directly from R
  # control <- rfeControl(functions=rfFuncs, method="repeatedcv", repeats=5)
  control <- rfeControl(functions=rfFuncs, method="cv", number=10)
  results <- rfe(X, Y, sizes=c(1:ncol(X)), rfeControl=control)
  print(results)
  predictors(results)
  plot(results, type=c("g", "o"))
}
# Based on RFE and Clusters looks like there are 3-4 core groups of important variables
# Top 5 Variables: DMC, DC, ISI, wind, FFMC
rfe_estimate(X, Y)

# Scale as a 9x9 GRID (w/colors)
# FireIndex (FWI) : ISI, BUI
# Does BUI highlight dangerous area, ISI velocity spread
