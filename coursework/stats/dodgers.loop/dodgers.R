# https://archive.ics.uci.edu/ml/datasets/Dodgers+Loop+Sensor
# Goal ---> Dodgers Loop Sensor Classification: Did a game occur at the stadium or not?
# Alternative Goal ---> Predict numCars within the precense of the Loop

##### Create Dodgers Sensor Reading and Game Event Data 
create_sensor_data <- function()
{
  Tab_data    <- read.csv("./Dodgers.data",  header=FALSE, sep=",", quote="\"", dec=".")
  rownames(Tab_data) <- 1:nrow(Tab_data)
  colnames(Tab_data) <- c("DateTime", "NumCars")
  # we only care about those observations for which we retrieved a valid sensor measurement 
  Tab_readings <- Tab_data[Tab_data$NumCars != -1, ]
  return (Tab_readings)
}

create_games_data <- function()
{
  Tab_data    <- read.csv("./Dodgers.events", header=FALSE, sep=",", quote="\"", dec=".")
  Tab_data$V6 <- iconv(Tab_data$V6, sub="")
  colnames(Tab_data) <- c("Date", "TimeBegin", "TimeEnd", "Attendance", "AwayTeam", "Score")
  Tab_data$Date <- as.Date(Tab_data$Date, format='%m/%d/%y')
  Tab_data$Day  <- weekdays(Tab_data$Date)
  return (Tab_data)
}

Tab_sensors_raw <- create_sensor_data()
Tab_events_raw  <- create_games_data()
def.par <- par(no.readonly = TRUE)

##### Feature Engineering: Create/Format Features
mns <- function(x){ return(x*60) }

create_sensor_features <- function(Tab_data, Tab_events_in)
{
  # normalize the Date to proper format and extract the day/Month
  # Tab_dodgers$NumCars   <- Tab_data[,2]
  # Tab_data$Date   <- as.Date( as.Date(Tab_readings$Date, format='%m/%d/%Y'), format='%m/%d/%y' )
  Tab_data$Date     <- as.Date(Tab_data$DateTime, format='%m/%d/%Y')
  Tab_data$Day      <- as.factor( weekdays(Tab_data$Date) )
  Tab_data$Month    <- as.factor( months(Tab_data$Date) )
  Tab_data$DateTime <- strptime(Tab_data$DateTime, "%m/%d/%Y %H:%M", tz="America/Los_Angeles" ) 
  Tab_data$TimeRec  <- format( as.POSIXlt(Tab_data$DateTime, format="%m/%d/%Y %H:%M"), "%H:%M:%S" )
  # Was it on a Gameday
  Tab_data$GameDay  <- ifelse( (Tab_data$Date %in% Tab_events_in$Date), 1, 0 )
  return (Tab_data)
}

create_game_features <- function(Tab_data, epsilon=60)
{
  Tab_data$GameStart <- strptime( paste(Tab_data$Date, Tab_data$TimeBegin),  "%Y-%m-%d %H:%M", tz="America/Los_Angeles" )
  Tab_data$GameEnd   <- strptime( paste(Tab_data$Date, Tab_data$TimeEnd),    "%Y-%m-%d %H:%M", tz="America/Los_Angeles" )
  # How long did the game last
  Tab_data$TimeElapseHrs <- as.numeric( strptime(Tab_data$TimeEnd,"%H:%M:%S") - 
                                        strptime(Tab_data$TimeBegin,"%H:%M:%S") )
  # Allow for additional time period before/after game starts/ends (default as 1 Hour (60min)) 
  Tab_data$GameStartEps <- Tab_data$GameStart - mns(epsilon)
  Tab_data$GameEndEps   <- Tab_data$GameEnd   + mns(epsilon)
  return (Tab_data)  
}

create_score_features <- function(Tab_data) 
{
  # How big was a Win or Loss, was it a a "Blowout" (Win-Loss Score > 5 Runs)
  Tab_scores <- do.call(rbind, strsplit(as.character(Tab_data$Score), " ") )
  Tab_scores <- data.frame(Tab_scores[,1], do.call(rbind, strsplit(as.character(Tab_scores[,2]), "-") ) )
  colnames(Tab_scores)  <- c("WinLoss", "WinScore", "LossScore")
  Tab_scores$Date       <- Tab_data$Date
  Tab_scores$WinScore   <- as.numeric(as.character(Tab_scores$WinScore))
  Tab_scores$LossScore  <- as.numeric(as.character(Tab_scores$LossScore))
  Tab_scores$DeltaScore <- Tab_scores$WinScore - Tab_scores$LossScore
  Tab_scores$Blowout    <- ifelse( (Tab_scores$DeltaScore >= 5 ), 1, 0)
  return (Tab_scores)
}
  
create_gameday_features <- function(Tab_sensors_in, Tab_events_in, Tab_scores_in)
{
  Tab_gameday <- Tab_sensors_in[Tab_sensors_in$GameDay == 1, ]
  # Was it while a Game is in Play
  Tab_gameday$Eventid    <- match(Tab_gameday$Date, Tab_events_in$Date)
  # During the sensor reading occur while Game is in session
  Tab_gameday$Inplay     <- as.numeric( 
  ( (Tab_events_in[match(Tab_gameday$Eventid, rownames(Tab_events_in)), ]$GameStart <= Tab_gameday$DateTime) & 
    (Tab_events_in[match(Tab_gameday$Eventid, rownames(Tab_events_in)), ]$GameEnd   >= Tab_gameday$DateTime) ) )
  # Was it Before/After a Certain Time (Eps) of the Game in Play: Epsilon=60 min
  Tab_gameday$InplayEps   <- as.numeric( 
  ( (Tab_events_in[match(Tab_gameday$Eventid, rownames(Tab_events_in)), ]$GameStartEps <= Tab_gameday$DateTime) & 
    (Tab_events_in[match(Tab_gameday$Eventid, rownames(Tab_events_in)), ]$GameEndEps   >= Tab_gameday$DateTime) ) )
  # Was the game related to a blowout causing the traffic to shift earlier
  Tab_gameday$Blowout     <-  as.numeric( Tab_scores_in[match(Tab_gameday$Eventid, rownames(Tab_scores_in)), c("Blowout")]  ) 
  # Was the game related to a Dodgers Loss, causing the traffic to shift earlier
  Tab_gameday$Loss        <-  as.numeric( Tab_scores_in[match(Tab_gameday$Eventid, rownames(Tab_scores_in)), c("WinLoss")] == "L" )
  # Was the game related to a large attendance, possibly allowing for more cars in the area (define category groups)
  evts  <- Tab_events_in[match(Tab_gameday$Eventid, rownames(Tab_events_in)), ] 
  games <- as.numeric( rownames(Tab_sensors_in) %in% rownames(Tab_gameday) )
  Tab_gameday$NumAttendance <- evts$Attendance
  Tab_gameday$Attendance[evts$Attendance < 40000 & evts$Attendance >= 30000] <- "30-40k"
  Tab_gameday$Attendance[evts$Attendance < 50000 & evts$Attendance >= 40000] <- "40-50k"
  Tab_gameday$Attendance[evts$Attendance < 60000 & evts$Attendance >= 50000] <- "50-60k"
  return (Tab_gameday)  
}

integrate_features <- function(Tab_sensors_in, Tab_gameday_in)
{
  Tab_sensors_in$GameInPlay    <- ifelse( (rownames(Tab_sensors_in) %in% rownames(Tab_gameday_in[Tab_gameday_in$Inplay==1, ])),1, 0)
  Tab_sensors_in$GameInPlayEps <- ifelse( (rownames(Tab_sensors_in) %in% rownames(Tab_gameday_in[Tab_gameday_in$InplayEps==1, ])),1, 0)
  
  Tab_sensors_in$GameBlowout   <- ifelse( (rownames(Tab_sensors_in) %in% rownames(Tab_gameday_in[Tab_gameday_in$Blowout==1, ])),1, 0)
  Tab_sensors_in$GameLoss      <- ifelse( (rownames(Tab_sensors_in) %in% rownames(Tab_gameday_in[Tab_gameday_in$Loss==1, ])),1, 0)
  
  games <- as.numeric( rownames(Tab_sensors_in) %in% rownames(Tab_gameday_in) )
  Tab_sensors_in$GameAttendance    <- as.factor( ifelse( games, Tab_gameday_in$Attendance, -1) )
  Tab_sensors_in$NumGameAttendance <- ifelse( games, Tab_gameday_in$NumAttendance,  -1)
  return (Tab_sensors_in)
}

Tab_events  <- create_game_features(Tab_events_raw)
Tab_scores  <- create_score_features(Tab_events)
# ordering applies (gameday in sensors must be created before gameday table)
Tab_sensors <- create_sensor_features(Tab_sensors_raw, Tab_events)
Tab_gameday <- create_gameday_features(Tab_sensors, Tab_events, Tab_scores)
Tab_sensors <- integrate_features(Tab_sensors, Tab_gameday)

##### EDA
library(ggplot2)
library(grid)
library(gridExtra)
plot_ts_numcars <- function(Tab_in, numcars_thresh, gameday_state=0, title_string="Gameday")
{
  date_range=c(as.Date ( strptime( paste("2005-04-10"),   "%Y-%m-%d", tz="America/Los_Angeles" )), 
               as.Date ( strptime( paste("2005-10-01"),  "%Y-%m-%d", tz="America/Los_Angeles" ) ) )
  
  Tab_day_readings <- Tab_in[Tab_in$GameDay==gameday_state, ]
  Tab_day_thresh   <- Tab_day_readings[Tab_day_readings$NumCars>numcars_thresh, ]
  pl <- ggplot(Tab_day_thresh, aes(Date, NumCars)) + geom_point(color="firebrick")
  pl <- pl+labs(x="Game Month", y="NumCars", 
                title=sprintf("NumCars Detected for %s Games", title_string))+xlim(date_range)  
  head(Tab_day_thresh[order(Tab_day_thresh$NumCars, decreasing=TRUE),,drop=FALSE])
  return (pl)
}

traffic_viz_games <- function(epsilon=0.15, thresh_hour=3.4, thresh_numcars=30)
{
  par(def.par)
  # No games in session this day
  p1 <- plot_ts_numcars(Tab_sensors, numcars_thresh=thresh_numcars, gameday_state=0, title_string="No Gameday")
  # Gamedays, Average Length of a Game
  avg <- mean(Tab_events$TimeElapseHrs)
  Tab_avg_readings <- Tab_sensors[Tab_sensors$Date %in% 
                      Tab_events[Tab_events$TimeElapseHrs >= (avg-epsilon) & 
                                 Tab_events$TimeElapseHrs <  (avg+epsilon), ]$Date, ]
  p2 <- plot_ts_numcars(Tab_avg_readings, numcars_thresh=thresh_numcars, gameday_state=1, title_string="Gameday ~Average")
  # Review observations for those games lasting >= 3.4 Hrs (Default), are these considered outliers?
  Tab_long_games  <- Tab_sensors[Tab_sensors$Date %in%  Tab_events[Tab_events$TimeElapseHrs >= thresh_hour, ]$Date, ]
  p3 <- plot_ts_numcars(Tab_long_games, numcars_thresh=thresh_numcars, gameday_state=1, title_string="Gameday Long")
  grid.arrange(p1, p2, p3, ncol=1)
}

traffic_viz_games_insession <- function(thresh=30)
{
  par(def.par)
  # No games in session this day
  p1 <- plot_ts_numcars(Tab_sensors, numcars_thresh=thresh, gameday_state=0, title_string="No Gameday")
  # Gamedays, while games are in play
  Tab_inplay <- Tab_sensors[Tab_sensors$GameInPlay==1, ]
  p2 <- plot_ts_numcars(Tab_inplay, numcars_thresh=thresh, gameday_state=1, title_string="In Session")
  # Gamedays, while games are in play +/- an epsilon threshold before/after the game (account for entering/exiting game)
  Tab_inplay_eps <- Tab_sensors[Tab_sensors$GameInPlayEps==1, ]
  p3 <- plot_ts_numcars(Tab_inplay_eps, numcars_thresh=thresh, gameday_state=1, title_string="In Session +/- Eps(t)=60 Min")
  grid.arrange(p1, p2, p3, ncol=1)
}

game_viz_stats <- function()
{
  par(def.par)
  date_range=c(as.Date ( strptime( paste("2005-04-10"),   "%Y-%m-%d", tz="America/Los_Angeles" )), 
               as.Date ( strptime( paste("2005-10-01"),  "%Y-%m-%d", tz="America/Los_Angeles" ) ) )
  
  # Most Games Occur on the weekends (Fri/Sat/Sun), although just as likely some occur on Tues/Wed
  print ("Games Occurences per Day:")
  print ( table(as.factor(Tab_events$Day)) )
  # Length of a game: Late July and August has some particular long games
  p_gl <- ggplot(Tab_events, aes(Date, TimeElapseHrs)) + geom_point(color="firebrick")
  p_gl <- p_gl+labs(x="Game Month", y="Hours", title="Game Length")+xlim(date_range)
  # Which time-period/games were blowout games with large margins (leaving early)
  Tab_blowouts <- Tab_scores[Tab_scores$Blowout==1,]
  p_bl <- ggplot(Tab_blowouts, aes(Date, DeltaScore)) + geom_point(color="firebrick")
  p_bl <- p_bl+labs(x="Game Month", y="Run Delta", title="Blowout (>=5 Runs) Games")+xlim(date_range)
  # Which games resulted in Losses, causing people to possibly leave early
  p_ls <- ggplot(Tab_scores[Tab_scores$WinLoss=="L",], aes(Date, WinLoss)) + geom_point(color="firebrick")
  p_ls <- p_ls+labs(x="Game Month", y="Loss", title="Dodger Game Losses")+xlim(date_range)
  grid.arrange(p_gl, p_bl, p_ls, ncol=1)
  #head(Tab_events[order(Tab_events$TimeElapseHrs, decreasing=TRUE),])
}

game_attendance_viz_stats <- function()
{
  par(def.par)
  date_range=c(as.Date ( strptime( paste("2005-04-10"),  "%Y-%m-%d", tz="America/Los_Angeles" )), 
               as.Date ( strptime( paste("2005-10-01"),  "%Y-%m-%d", tz="America/Los_Angeles" ) ) )

  # Attendance per a function of the game  
  p_at <- ggplot(Tab_events, aes(Date, Attendance)) + geom_point(color="firebrick")
  p_at <- p_at+labs(x="Game Month", y="Attendance", title="Game Attendance")+xlim(date_range)
  # Attendance as a function of the numCars
  p_nc_at <- ggplot(Tab_gameday, aes(NumAttendance, NumCars)) + geom_point(color="firebrick")
  p_nc_at <- p_nc_at+labs(x="Attendance per Game", y="NumCars", title="NumCars detected as function of Game Attendance")
  grid.arrange(p_at, p_nc_at, ncol=1)
  #head(Tab_gameday[order(Tab_gameday$Attendance, decreasing=TRUE),,drop=FALSE])
} 

game_viz_stats()
game_attendance_viz_stats()
traffic_viz_games()
traffic_viz_games_insession()

##### More Metrics/Visualizations
# Patterns of Interest (Standard Deviations)
# Pattern Window

##### Outliers: TBD
##### Sampling: Take a small sample to begin with (size=5000)
# This may not be the best way to handle time series sampling (e.g. createTimeSlices)
set.seed(12202015)
Tab_sel <- Tab_sensors[sample(1:nrow(Tab_sensors), 5000,replace=FALSE),]
##### Select Features (let the target be determining if a game is in Play 1 hour before/after the event)
Tab_sel <- Tab_sel[, c("Day", "Month", "NumCars", "GameDay", "GameInPlay", "GameInPlayEps",
                       "GameBlowout", "GameLoss", "NumGameAttendance", "GameAttendance")]
##### Normalization: TBD (mu=0, sigma=1)
##### Scale values (min-max)
Tab_sel$NumGameAttendance <- (Tab_sel$NumGameAttendance - min(Tab_sel$NumGameAttendance))/
                             (max(Tab_sel$NumGameAttendance) - min(Tab_sel$NumGameAttendance))
##### Train/Test Split (per time-series based data)
library(caret)
partition      <- createDataPartition(y=Tab_sel$GameInPlayEps, p=0.70, list=FALSE)
training_part  <- Tab_sel[partition, ]
testing_part   <- Tab_sel[-partition, ]
# Using the Train data, resplit into training part and validation part
partition_tr        <- createDataPartition(y=training_part[, c("GameInPlayEps")], p=0.70, list=FALSE)
training_part_sub   <- training_part[partition_tr, ]
validation_part_sub <- training_part[-partition_tr, ]

X_tr <- training_part_sub[, !colnames(training_part_sub) %in% c("GameInPlayEps")]
Y_tr <- training_part_sub[, c("GameInPlayEps"), drop=FALSE]
X_val <- validation_part_sub[, !colnames(validation_part_sub) %in% c("GameInPlayEps")]
Y_val <- validation_part_sub[, c("GameInPlayEps"), drop=FALSE]
X_ts  <- testing_part[, !colnames(testing_part) %in% c("GameInPlayEps")]
Y_ts  <- testing_part[, c("GameInPlayEps"), drop=FALSE]

##### Numerical Correlation to Target(GameInPlayEps) on Entire Sample
# Most Correlated with output (ordered): GameInPlay, GameDay, NumGameAttendance, GameLoss, NumCars, GameBlowout
cor(Tab_sel[, c("NumCars", "GameDay", "GameInPlay", "GameBlowout", "GameLoss", "NumGameAttendance", "GameInPlayEps")])

##### RFE
# Top Variables: GameInPlay, NumCars, Day, NumGameAttendance, Month
rfe_estimate <- function(X_in, Y_in)
{
  # verify some data with RFE directly from R
  control <- rfeControl(functions=rfFuncs, method="cv", number=10)
  results <- rfe(X_in, Y_in, sizes=c(1:ncol(X_in)), rfeControl=control)
  print(results)
  predictors(results)
  plot(results, type=c("g", "o"))
}
rfe_estimate(X_tr, Y_tr$GameInPlayEps)

##### Perform prediction via random forest
library(randomForest)
model_importance <- function(training_sub_in, target_sub_in, metric_type)
{
  model_base <- randomForest(as.factor(target_sub_in$GameInPlayEps)~., 
                             data = training_sub_in, 
                             importance=TRUE, ntree=200)
  
  metrics   <- as.data.frame(importance(model_base, metric_type))
  # plot the importance values
  varImpPlot(model_base, scale=TRUE)
  return (metrics)
}
# Model Importance based on GINI Index wihtout Transformation of PCA: 
# GameInPlay, NumCars, NumGameAttendance, Day, GameAttendance
metric_importance <- model_importance(X_tr, Y_tr, metric_type=2)
metric_importance <- metric_importance[order(metric_importance, decreasing=TRUE),,drop=FALSE]
labels <- rownames(metric_importance)[1:5]
cat("Top 5 per Feature Importance: ", labels)

##### Dimensionality Reduction: PCA
library(FactoMineR)
eval_pca <- function(X_in, cat_col)
{
  par(def.par)
  # if scale.unit = FALSE -> a covariance matrix is used instead of a correlation matrix
  # currently selecting scale.unit = TRUE to normalize the data
  pca <- PCA(X_in, graph = TRUE, scale.unit = TRUE, quali.sup=cat_col)
  # eignvalue < 1 is commonly used as cutoff (PCs account for more variance than in original input) -> 3
  p1 <- ggplot(pca$eig, aes(c(1:length(eigenvalue)), eigenvalue)) + 
        geom_point(color="firebrick") + geom_line()
  p1 <- p1+labs(x="component dimensions", y="eignevalue", title="PCA Scree Plot")
  # alternatively look at the variance to evaluate maximized variance ~ 95% (2*sigma)
  p2 <- ggplot(pca$eig, aes(c(1:length(eigenvalue)), `cumulative percentage of variance`)) + 
        geom_point(color="firebrick") + geom_line()
  p2 <- p2+labs(x="component dimensions", y="cumulative percentage of variance", title="PCA Scree Plot")
  p3 <- ggplot(pca$eig, aes(c(1:length(eigenvalue)), `percentage of variance`)) + 
        geom_point(color="firebrick") +geom_line()
  p3 <- p3+labs(x="component dimensions", y="percentage of variance", title="PCA Scree Plot")
  # plot directions of features
  # plot 5 most contribution variables
  # plot(pca,select="contrib 5")
  # plot categorical related variables
  # plotellipses(pca_o,habillage=9)
  
  grid.arrange(p1, p2, p3, ncol=3)
  return (pca)
  #print(pca$eig)
  #print(pca$var$coord)
  #head(pca$ind$coord)
}
# With 5 Components for this sample, we have > 95% Variance (2*sigma)
# Looks like there are 2 distinct clusters forming (but this is due to no games being attended = -1)
pca_o <- eval_pca(X_tr,  c(1,2,9))
summary(pca_o)
dimdesc(pca_o, axes = 1:2)


##### TBD: Use the transformed dimension space from PCA Components (R^n)->(R^k) {n=9,k=4}
##### Fit into prediction model


