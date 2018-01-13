# load data
setwd("~/Desktop/Ds-Coursera/05-practicalML/Human-Activity-Recognition")
train.data <- read.csv("pml-training.csv")
dim(train.data) #19622   160
test.data <- read.csv("pml-testing.csv")
dim(test.data) # 20 160

# load libraries
library(caret)
#===========================================
# PreProcessing steps
# --------------------
# partition data
parts.train <- createDataPartition(train.data$classe, p=0.70, list=FALSE)
train.part <- train.data[parts.train,] 
dim(train.part) # 13737   160
test.part <- train.data[-parts.train,]
dim(test.part) #5885  160

# lets clean the sets, by removing NAs
# using Near zero variance "NZV"
nzv <- nearZeroVar(train.data)
train.part <- train.part[,-nzv]
test.part <- test.part[,-nzv]

# lets check dimensions again
dim(train.part) # 13737   100
dim(test.part) # 13737   100

# remove variables that are mostly NA
NAs <- sapply(train.part, function(x) mean(is.na(x))) > 0.95
train.part <- train.part[, NAs==FALSE]
test.part  <- test.part[, NAs==FALSE]

# lets check dimensions again
dim(train.part) # 13737    59 
dim(test.part) # 13737   59

# remove 1st 5 columns as all not importatnt features 
train.part <- train.part[,-(1:5)] # 13737    54
test.part <- test.part[,-(1:5)] # 13737    54

#===========================================
# Lets do some Analysis
#-----------------------
library(corrplot)

# lets check the corelation matrix
corMatrix <- cor(train.part[, -54])

# save figure
png('corr-plot.png')
corr.fig <- corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
# darker colores means high correlation between variables
dev.off()

#===========================================
# Lets start building models
#---------------------------
# Three methods will be applied to model the regressions (in the Train dataset) 
# and the best one (with higher accuracy when applied to the Test dataset)

## M1 : Random forest
#--------------------
library(randomForest)
# model fit
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=train.part, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel

#Lets do prediction on Test dataset
predict.RF <- predict(modFitRandForest, newdata=test.part)
conf.RF <- confusionMatrix(predict.RF, test.part$classe)
conf.RF

# save figure
png('conf-RF.png')
# plot matrix results
plot(conf.RF$table, col = conf.RF$byClass
     , main = paste("Random Forest (Accuracy) =", round(conf.RF$overall['Accuracy'], 4)))
dev.off()

## M2 : Decision Trees
#----------------------
library(rpart)
library(rpart.plot)
library(rattle)
library(knitr)

# model fit
set.seed(12345)
modFit.DT <- rpart(classe ~ ., data=train.part, method="class")
# plot the classification tree
#rattle::fancyRpartPlot(modFit.DT$finalModel)
fancyRpartPlot(modFit.DT)

# prediction on Test dataset
predict.DT <- predict(modFit.DT, newdata=test.part, type="class")
conf.DT <- confusionMatrix(predict.DT, test.part$classe)
conf.DT

# save figure
png('conf-DT.png')
# plot matrix results
plot(conf.DT$table, col = conf.DT$byClass, 
     main = paste("Decision Tree (Accuracy) =",
                  round(conf.DT$overall['Accuracy'], 4)))
dev.off()

## M3 : Generalized Boosted Model
#---------------------------------
library(gbm)
library(plyr)
# model fit
set.seed(12345)
control.boost <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFit.boost  <- train(classe ~ ., data=train.part, method = "gbm",
                    trControl = control.boost, verbose = FALSE)
modFit.boost$finalModel

# prediction on Test dataset
predict.boost <- predict(modFit.boost, newdata=test.part)
conf.boost <- confusionMatrix(predict.boost, test.part$classe)
conf.boost

# save figure
png('conf-Boost.png')
# plot matrix results
plot(conf.boost$table, col = conf.boost$byClass, 
     main = paste("Boosting (Accuracy) =", round(conf.boost$overall['Accuracy'], 4)))
dev.off()

# ===========================================
# RF vs DT vs Boosting
#----------------------
# Random Forest : 0.9968
# Decision Tree : 0.7412
# Genl Boosting : 0.9839

# Then Random forest WIN ;)
# lets use it on real testing data
predict.real.test <- predict(modFitRandForest, newdata=test.data)
predict.real.test

# Final output
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E
