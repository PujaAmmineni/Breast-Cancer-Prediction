# Load required libraries
library(caret)
library(corrplot)
library(caret)
library(e1071)
library(corrplot)
library(pROC)
library(glmnet)
library(kernlab)
library(nnet)
library(MASS)
library(earth)

# Load dataset
wdbc <- read.csv("C:\\Users\\Puja\\Documents\\Data Files\\wdbc.data", header = TRUE)
wdbc <- wdbc[, -1] # Remove ID column

# Select numeric columns
numeric_wdbc <- wdbc[, sapply(wdbc, is.numeric)]

# Compute correlation and remove highly correlated predictors
correlations <- cor(numeric_wdbc)
corrplot(correlations, order = "hclust")
highCorr <- findCorrelation(correlations, cutoff = 0.85)
filtered_data <- numeric_wdbc[, -highCorr]
length(highCorr)

# Preprocessing: Center and scale data
filtered_preprocess <- preProcess(filtered_data, method = c("center", "scale"))
scaled_filtered <- predict(filtered_preprocess, filtered_data)

# Add Diagnosis column back
filtered_dataset <- data.frame(scaled_filtered, Diagnosis = wdbc$Diagnosis)

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(filtered_dataset$Diagnosis, p = 0.8, list = FALSE)
filtered_train <- filtered_dataset[trainIndex, ]
filtered_test <- filtered_dataset[-trainIndex, ]
length(filtered_train$area1)
length(filtered_test$area1)

train_control <- ctrl <- trainControl(method = "repeatedcv",
                                      number = 10,
                                      repeats = 5,
                                      summaryFunction = twoClassSummary,
                                      classProbs = TRUE,
                                      savePredictions = TRUE)
# Logistic Regression
logistic_model <- train(
  x = filtered_train[, -ncol(filtered_train)], y = filtered_train$Diagnosis,
  method = "glm",
  metric = "ROC",
  trControl = train_control
)
print(logistic_model)
summary(logistic_model)
#plot(logistic_model)
filtered_test$Diagnosis <- factor(filtered_test$Diagnosis, levels = c("B", "M"))
predicted_labels <- predict(logistic_model, newdata = filtered_test[, -ncol(filtered_test)])
predicted_probs <- predict(logistic_model, newdata = filtered_test[, -ncol(filtered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(filtered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(filtered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - Logistic Regression", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc

# LDA
lda_model <- train(
  x = filtered_train[, -ncol(filtered_train)], y = filtered_train$Diagnosis,
  method = "lda",
  metric = "ROC",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
)
print(lda_model)
#plot(lda_model)
summary(lda_model)
filtered_test$Diagnosis <- factor(filtered_test$Diagnosis, levels = c("B", "M"))
predicted_labels <- predict(lda_model, newdata = filtered_test[, -ncol(filtered_test)])
predicted_probs <- predict(lda_model, newdata = filtered_test[, -ncol(filtered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(filtered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(filtered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - LDA", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)


# QDA
qda_model <- train(
  x = filtered_train[, -ncol(filtered_train)], y = filtered_train$Diagnosis,
  method = "qda",
  metric = "ROC",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
)
print(qda_model)
#plot(qda_model)
summary(qda_model)
filtered_test$Diagnosis <- factor(filtered_test$Diagnosis, levels = c("B", "M"))
predicted_labels <- predict(qda_model, newdata = filtered_test[, -ncol(filtered_test)])
predicted_probs <- predict(qda_model, newdata = filtered_test[, -ncol(filtered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(filtered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(filtered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - QDA", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)



#fda

fda_model <- train(
  x = filtered_train[, -ncol(filtered_train)], y = filtered_train$Diagnosis,
  method = "fda",
  tuneGrid = expand.grid(.degree = 1:2, .nprune = 2:15),
  metric = "ROC",
  preProc = c("center", "scale"),
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
)
fda_model
plot(fda_model)
filtered_test$Diagnosis <- factor(filtered_test$Diagnosis, levels = c("B", "M"))
predicted_labels <- predict(fda_model, newdata = filtered_test[, -ncol(filtered_test)])
predicted_probs <- predict(fda_model, newdata = filtered_test[, -ncol(filtered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(filtered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(filtered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - FDA", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)




#knn
knn_model <- train(
  x = filtered_train[, -ncol(filtered_train)], y = filtered_train$Diagnosis,
  method = "knn",
  tuneGrid = data.frame(.k = 1:20),
  metric = "ROC",
  preProc = c("center", "scale"),
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
)
knn_model 
plot(knn_model)
predicted_labels <- predict(knn_model, newdata = filtered_test[, -ncol(filtered_test)])
predicted_probs <- predict(knn_model, newdata = filtered_test[, -ncol(filtered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(filtered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(filtered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - KNN", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)



# Regularized Discriminant Analysis (RDA)
rda_model <- train(
  x = filtered_train[, -ncol(filtered_train)], 
  y = filtered_train$Diagnosis,
  method = "rda",
  tuneGrid = expand.grid(.gamma = seq(0, 1, length = 10), .lambda = seq(0, 1, length = 15)), 
  metric = "ROC",
  trControl = trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)
)
print(rda_model)
plot(rda_model)
summary(rda_model)
filtered_test$Diagnosis <- factor(filtered_test$Diagnosis, levels = c("B", "M"))
predicted_labels <- predict(rda_model, newdata = filtered_test[, -ncol(filtered_test)])
predicted_probs <- predict(rda_model, newdata = filtered_test[, -ncol(filtered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(filtered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(filtered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - RDA", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)



# Mixture Discriminant Analysis (MDA)
mda_model <- train(
  x = filtered_train[, -ncol(filtered_train)], 
  y = filtered_train$Diagnosis,
  method = "mda",
  tuneGrid = expand.grid(.subclasses = seq(1, 10, by = 1)), 
  metric = "ROC",
  trControl = trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)
)
print(mda_model)
plot(mda_model)
summary(mda_model)
filtered_test$Diagnosis <- factor(filtered_test$Diagnosis, levels = c("B", "M"))
predicted_labels <- predict(mda_model, newdata = filtered_test[, -ncol(filtered_test)])
predicted_probs <- predict(mda_model, newdata = filtered_test[, -ncol(filtered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(filtered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(filtered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - MDA", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)


