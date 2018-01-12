# load data
setwd("~/Desktop/Ds-Coursera/05-practicalML/Human-Activity-Recognition")
train.data <- read.csv("pml-training.csv")
dim(train.data) #19622   160
test.data <- read.csv("pml-testing.csv")
dim(test.data) # 20 160

# load libraries
library(caret)

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


