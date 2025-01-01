# Load required libraries
library(caret)
library(corrplot)
wdbc <- read.csv("C:\\Users\\Puja\\Desktop\\Predictive\\Data\\wdbc.data", header = TRUE)
wdbc

wdbc <- wdbc[, -1] 


numeric_wdbc <- wdbc[, sapply(wdbc, is.numeric)] 
diagnosis <- wdbc$Diagnosis 
correlations <- cor(numeric_wdbc)
corrplot(correlations, order = "hclust", main = "Correlation Matrix")
length(numeric_wdbc)

unfiltered_preprocess <- preProcess(numeric_wdbc, method = c("center", "scale"))
scaled_unfiltered <- predict(unfiltered_preprocess, numeric_wdbc)

unfiltered_dataset <- data.frame(scaled_unfiltered, Diagnosis = diagnosis)

unfiltered_dataset$Diagnosis <- factor(unfiltered_dataset$Diagnosis, levels = c("B", "M"))



set.seed(123)
trainIndex <- createDataPartition(unfiltered_dataset$Diagnosis, p = 0.8, list = FALSE)
unfiltered_train <- unfiltered_dataset[trainIndex, ]
unfiltered_test <- unfiltered_dataset[-trainIndex, ]

#plsda
pls_model <- train(
  x = unfiltered_train[, -ncol(unfiltered_train)], y = unfiltered_train$Diagnosis,
  method = "pls",
  tuneGrid = expand.grid(.ncomp = 1:15),
  metric = "ROC",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
)
pls_model
plot(pls_model)
predicted_labels <- predict(pls_model, newdata = unfiltered_test[, -ncol(unfiltered_test)])
predicted_probs <- predict(pls_model, newdata = unfiltered_test[, -ncol(unfiltered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(unfiltered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(unfiltered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - PLSDA", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)

#penalized models
glmnet_model <- train(
  x = unfiltered_train[, -ncol(unfiltered_train)], y = unfiltered_train$Diagnosis,
  method = "glmnet",
  tuneGrid = expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), .lambda = seq(.01, .2, length = 10)),
  metric = "ROC",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
)
glmnet_model
plot(glmnet_model)
predicted_labels <- predict(glmnet_model, newdata = unfiltered_test[, -ncol(unfiltered_test)])
predicted_probs <- predict(glmnet_model, newdata = unfiltered_test[, -ncol(unfiltered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(unfiltered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(unfiltered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - Penalized Logistic Regression", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)




#svm
svm_model <- train(
  x = unfiltered_train[, -ncol(unfiltered_train)], y = unfiltered_train$Diagnosis,
  method = "svmRadial",
  tuneGrid = expand.grid(.sigma = sigest(as.matrix(unfiltered_train[, -ncol(unfiltered_train)]))[1], .C = 2^(seq(-4, 4))),
  metric = "ROC",
  preProc = c("center", "scale"),
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
)
svm_model
plot(svm_model)
predicted_labels <- predict(svm_model, newdata = unfiltered_test[, -ncol(unfiltered_test)])
predicted_probs <- predict(svm_model, newdata = unfiltered_test[, -ncol(unfiltered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(unfiltered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc$auc
roc_auc <- roc(unfiltered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - SVM", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)

varImp(svm_model)

# Naive Bayes
nb_model <- train(
  x = unfiltered_train[, -ncol(unfiltered_train)], 
  y = unfiltered_train$Diagnosis,
  method = "nb",
  tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
  metric = "ROC",
  trControl = trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)
)
print(nb_model)
plot(nb_model)
predicted_labels <- predict(nb_model, newdata = unfiltered_test[, -ncol(unfiltered_test)])
predicted_probs <- predict(nb_model, newdata = unfiltered_test[, -ncol(unfiltered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(unfiltered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(unfiltered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - Naive Bayes", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)


# Neural Network
nn_model <- train(
  x = unfiltered_train[, -ncol(unfiltered_train)], 
  y = unfiltered_train$Diagnosis,
  method = "nnet",
  tuneGrid = expand.grid(.size = 1:9, .decay = c(0, .1, 1, 2)),
  metric = "ROC",
  preProc = c("center", "scale"), 
  trControl = trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary),
  trace = FALSE
)

print(nn_model)
plot(nn_model)
predicted_labels <- predict(nn_model, newdata = unfiltered_test[, -ncol(unfiltered_test)])
predicted_probs <- predict(nn_model, newdata = unfiltered_test[, -ncol(unfiltered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(unfiltered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc$auc
roc_auc <- roc(unfiltered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - Neural Network", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)

