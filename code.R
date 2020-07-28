# Load all required libraries

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gbm)) install.packages("gbm")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(xgboost)) install.packages("xgboost")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(PRROC)) install.packages("PRROC")
if(!require(reshape2)) install.packages("reshape2")

library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(gbm)
library(caret)
library(xgboost)
library(e1071)
library(class)
library(ROCR)
library(randomForest)
library(PRROC)
library(reshape2)


# set wd, download and load the data set
WD <- getwd()
WD
if(!is.null(WD)) setwd("~/credit-card-fraud-detection/")

dl <- tempfile()
download.file("https://www.kaggle.com/mlg-ulb/creditcardfraud/download/creditcard.zip", dl)

unzip(dl,"")
download.file(url, "data/creditcard.csv")
creditcard <- read_csv("data/creditcard.csv")

head(creditcard)
dim(creditcard)

# Imbalanced target
creditcard %>% group_by(Class) %>% summarise(Count = n())

# Fraud Total Amount 
creditcard[creditcard$Class == 1,] %>% group_by(Amount) %>% sum()

# Fraud amount propotion
Fraudamount = creditcard[creditcard$Class == 1,] %>% group_by(Amount) %>% sum()

nonamount = creditcard[creditcard$Class == 0,] %>% group_by(Amount) %>% sum()

Fraudamount / nonamount


# Missing Values
sum(is.na(creditcard))

# Header of dataset
head(creditcard)

# Distribution of Data
creditcard[creditcard$Class == 1,] %>%
  ggplot(aes(Time)) + theme_minimal()  + geom_histogram() +
  labs(title = "Distribution of Frauds along time", x = "Time", y = "Frequency") +
  theme_economist()

creditcard[creditcard$Class == 1,] %>%
  ggplot(aes(Amount)) + theme_minimal()  + geom_histogram(binwidth = 40) +
  labs(title = "Frauds Amounts Distributions", x = "Amount in dollars", y = "Frequency") +
  theme_economist()

# Study Correlations

cormat <- round(cor(creditcard),2)
melted_cormat <- melt(cormat)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") 

## Data Preparation

# Convert target variable class into factor (categorical)
creditcard$Class <- as.factor(creditcard$Class)


# Train- test split
train_index <- createDataPartition(y = creditcard$Class, p = 0.6, list = FALSE)

train <- creditcard[train_index,]
test_cv <- creditcard[-train_index,]

test_index <- createDataPartition(y = test_cv$Class, p = .5, list = FALSE)

test <- test_cv[test_index,]
cv <- test_cv[-test_index,]

rm(train_index, test_index, test_cv)


## Model Building

# Random forrest
rf_model <- randomForest(Class ~ ., data = train, ntree = 500)

# Get the feature importance
feature_imp_rf <- data.frame(importance(rf_model))

# Make predictions based on this model
predictions <- predict(rf_model, newdata=test)

# Compute the AUC and AUPCR
pred <- prediction(as.numeric(as.character(predictions)), as.numeric(as.character(test$Class)))

auc_val_rf <- performance(pred, "auc")

auc_plot_rf <- performance(pred, 'sens', 'spec')

aucpr_plot_rf <- performance(pred, "prec", "rec", curve = T,  dg.compute = T)

aucpr_val_rf <- pr.curve(scores.class0 = predictions[test$Class == 1], scores.class1 = predictions[test$Class == 0],curve = T,  dg.compute = T)

# make the relative plot
plot(auc_plot_rf, main=paste("AUC:", auc_val_rf@y.values[[1]]))

# Adding the respective metrics to the results dataset

results <- data.frame(Model = "Random Forest", AUC = auc_val_rf@y.values[[1]], AUCPR = aucpr_val_rf$auc.integral)

results

### KNN - K-Nearest Neighbors
# Build a KNN Model with Class as Target and all other
# variables as predictors. k is set to 5

knn_model <- knn(train[,-30], test[,-30], train$Class, k=5, prob = TRUE)

# Compute the AUC and AUCPR for the KNN Model
pred <- prediction(as.numeric(as.character(knn_model)), as.numeric(as.character(test$Class)))

auc_val_knn <- performance(pred, "auc")

auc_plot_knn <- performance(pred, 'sens', 'spec')
aucpr_plot_knn <- performance(pred, "prec", "rec")

aucpr_val_knn <- pr.curve(
  scores.class0 = knn_model[test$Class == 1], scores.class1 = knn_model[test$Class == 0], curve = T, dg.compute = T)

# Make the relative plot
plot(aucpr_val_knn)


# Adding the respective metrics to the results dataset
results <- results %>% add_row(Model = "K-Nearest Neighbors k=5", AUC = auc_val_knn@y.values[[1]], AUCPR = aucpr_val_knn$auc.integral)
results

#### SVM - Support Vector Machine
# Build a SVM Model with Class as Target and all other
# variables as predictors. The kernel is set to sigmoid

svm_model <- svm(Class ~ ., data = train, kernel='sigmoid')

# Make predictions based on this model
predictions <- predict(svm_model, newdata=test)

# Compute AUC and AUCPR
pred <- prediction(as.numeric(as.character(predictions)), as.numeric(as.character(test$Class)))

auc_val_svm <- performance(pred, "auc")

auc_plot_svm <- performance(pred, 'sens', 'spec')
aucpr_plot_svm <- performance(pred, "prec", "rec")

aucpr_val_svm <- pr.curve(
  scores.class0 = predictions[test$Class == 1], scores.class1 = predictions[test$Class == 0], curve = T, dg.compute = T)

# Make the relative plot
plot(aucpr_val_svm)


# Adding the respective metrics to the results dataset
results <- results %>% add_row(Model = "SVM - Support Vector Machine", AUC = auc_val_svm@y.values[[1]], AUCPR = aucpr_val_svm$auc.integral)
results

# Results
results