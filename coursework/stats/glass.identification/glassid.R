# Datasets
# https://archive.ics.uci.edu/ml/datasets/Glass+Identification

library(FactoMineR)
# Id: 1-214
# Weight Percent in oxide
# Y: {1-7}
TabGData    <- read.csv("./glass.data.txt", header=FALSE, sep=",", quote="\"", dec=".")
colnames(TabGData) <- c("Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "GlassType")
ynames <- c("building_windows_float_processed", "building_windows_non_float_processed",
            "vehicle_windows_float_processed", "vehicle_windows_non_float_processed",
            "containers", "tableware", "headlamps")
DCor <- cor(TabGData)
# Num NA Observations
nmissing <- TabGData[is.na(TabGData), ]
# mapping
X <- TabGData[, 1:10]
Y <- TabGData[, 11]
summary(X)
# Train/Split

# RFE
library(caret);
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(X, Y, sizes=c(1:ncol(X)), rfeControl=control)
print(results)
predictors(results)
plot(results, type=c("g", "o"))

# Min-Max Scaling
# Plots
plot(TabGData$Id, TabGData$GlassType)
plot(TabGData$RI, TabGData$GlassType)
plot(TabGData$Na, TabGData$GlassType)

# Featurization

# estim_ncp to calculate best number of dimensions
estim_ncp(TabGData[, 3:10], ncp.min=0, ncp.max = 10, scale=TRUE, method="GCV")

# PCA Methods
glass.pca <- PCA(TabGData[, 3:10], graph = TRUE, scale.unit = FALSE)
glass.pca$eig
glass.pca$var$coord
head(glass.pca$ind$coord)
plot(c(1:8), glass.pca$eig$`percentage of variance`)
# check correlation (which features are mostly correlated with which component)
dimdesc(glass.pca)

# PCA 
filt <- c("Mg", "Ca", "K", "Ba")
glass.pca <- PCA(TabGData[, filt], graph = TRUE, scale.unit = FALSE)
glass.pca$eig
glass.pca$var$coord
head(glass.pca$ind$coord)

# via princomp in stats
glass.pcacomp <- princomp(TabGData[, 3:10], cor=TRUE) 
glass.pcacomp$sdev
unclass(glass.pcacomp$loadings)
head(glass.pcacomp$scores)


