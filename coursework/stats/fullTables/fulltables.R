##### Load files and Create Tables
TabCarbo  <- read.csv("./AreaWksAllcarbo.txt")
TabCarbof <- TabCarbo[-31, ]
TabEmpty  <- read.csv("./AreaWksAllEmpty.txt")
TabEmptyf <- TabEmpty[-31, ]
TabGlass  <- read.csv("./AreaWksAllglass.txt", header=TRUE, sep="\t", quote="\"", dec=".")
TabGlassf <- TabGlass[-31, ]
TabMelam  <- read.csv("./AreaWksAllmelam.txt", header=TRUE, sep="\t", quote="\"", dec=".")
TabMelamf <- TabMelam[-31, ]
TabWater  <- read.csv("./AreaWksAllWater.txt", header=TRUE, sep="\t", quote="\"", dec=".")
TabWaterf <- TabWater[-31, ]
# create data table
TableAll  <- rbind(TabCarbof, TabEmptyf, TabGlassf, TabMelamf, TabWaterf)
X <- TableAll[, c("RATIO21", "RATIO31")]
colnames(X) <- c("x", "y")
##### create clusters centers based on mean
C1 <- c(mean(TabCarbof[, c("RATIO21")]), mean(TabCarbof[, c("RATIO31")]))
C2 <- c(mean(TabEmptyf[, c("RATIO21")]), mean(TabEmptyf[, c("RATIO31")]))
C3 <- c(mean(TabGlassf[, c("RATIO21")]), mean(TabGlassf[, c("RATIO31")]))
C4 <- c(mean(TabMelamf[, c("RATIO21")]), mean(TabMelamf[, c("RATIO31")]))
C5 <- c(mean(TabWaterf[, c("RATIO21")]), mean(TabWaterf[, c("RATIO31")]))
center <- t(data.frame(C1,C2,C3,C4,C5))
# perform cluster via kmeans (with outliers), create plots for visualization
km <- kmeans(X, centers=center, iter.max=100, nstart=10,
             algorithm = c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"), trace=FALSE)
plot(X, col=km$cluster)
points(km$centers, col = 1:2, pch = 8, cex = 2)
# alternative method of plot visualization
# plot all points elipses
library(cluster)
library(fpc)
clusplot(X, km$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
# plot numerical version
plotcluster(X, km$cluster)


##### filter outliers
# manually remove distance points for each centroid and keep 10 best points
filter_outliers <- function(df_in)
{
  for(i in 30:11)
  {
    xhat <- mean(df_in[, c("x")])  
    yhat <- mean(df_in[, c("y")])
    df_in$dx <- abs(xhat - df_in[, c("x")])
    df_in$dy <- abs(yhat - df_in[, c("y")])
    df_in$d  <- sqrt( (df_in$dx)**2 + (df_in$dy)**2)
    df_in    <- df_in[order(df_in$d, decreasing=TRUE), ]
    df_in    <- df_in[-1,]
  }
  return(df_in)
}

exec_outliers <- function(table_in, partition_size=30)
{
  df <- data.frame()
  for (i in seq(1,150, by=30) )
  {
    out <- as.data.frame( filter_outliers(table_in[i:(i+(30-1)),]) )
    out$cluster <- toString(round(i/30)+1)
    df <- rbind(df, out)
  }
  return (df)
}

clustered_filtered_outliers <- exec_outliers(X)
print(clustered_filtered_outliers)

##### Alternative Packages: PAM(median,mode), Clara(for >200 observations) Package
# install.packages("fpc")
# install.packages("cluster")
library(fpc)
library(cluster)
set.seed(12152015)
# direct pam usage via input Complete Filtered table
Xprime <- TableAll[, c("RATIO21", "RATIO31")]
colnames(Xprime) <- c("x", "y")
pm <- pamk(Xprime, 5)
cat("Num Clusters: ", pm$nc)
pm$pamobject$medoids
plot(Xprime, col=pm$pamobject$clustering)
points(pm$pamobject$medoids, col = 1:2, pch = 8, cex = 2)
# pam usage based on pairwise dissimilarity matrix
Xdist <- daisy(Xprime, metric="euclidean", stand=FALSE)
Xdist <- as.matrix(Xdist)
pm_dist <- pamk(Xdist, 5)
cat("Num Clusters: ", pm_dist$nc)
#pm_dist$pamobject$medoids
plot(Xdist, col=pm_dist$pamobject$clustering)
points(pm_dist$pamobject$medoids, col = 1:2, pch = 8, cex = 2)
# Avoid loop by using PAM
# Method: use lapply method over a function sequence per a table for vectorized functions

##### Alternative Clustering Methods
# optics
library(dbscan)
res <- optics(X, eps =3, minPts = 10)
par(bg="white")
plot(res)
res <- optics_cut(res, eps = 0.05)
plot(res)
# dbscan
library(fpc)
par(bg="white")
res <- dbscan(X, eps=0.08, MinPts = 10)
plot(X, col=res$cluster +1)
kNNdistplot(X, k=5)


##### find the elbow for optimal k
vark <- list()
k = floor(sqrt(nrow(X)/2))
for (i in 1:k)
{
  kx <- kmeans(X, i, iter.max=100, nstart=1, algorithm="Hartigan-Wong", trace=FALSE )
  vark[i]  <- sum(kx$withinss)
  #vark[i]  <- kx$tot.withinss
}
plot(c(1:k), vark)
# plot the percentage of variance explained by the clusters against the number of clusters
# http://stackoverflow.com/questions/6645895/calculating-the-percentage-of-variance-measure-for-k-means





