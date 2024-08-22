### CS699 Final Project
## Brian Ramm
## Yidan Sun

# import packages
library(dplyr)
library(FSelector)
library(Boruta)
library(RWeka)
library(e1071)
library(caret)
library(rsample)
library(tidyverse)
library(C50)
library(rpart)
library(rpart.plot)
library(pROC)
library(MASS)
library(ROSE)
library(stats)
library(randomForest)
library(mltools)
library(psych)
library(glmnet)
library(kernlab)
library(prediction)


setwd("E:\\BU\\2024spring\\CS 699\\project\\Final_Report")
df = read.csv("cleaned_trimmed_data.csv")

# Class: N for 3926, Y for 925: need to Oversample of Y or Undersample N
table(df$Class)
set.seed(123)

# Split the dataset into training and testing sets
train_index <- createDataPartition(y = df$Class, p = 0.8, list = FALSE)

# Create training and testing sets
train_data <- df[train_index, ]
test_data <- df[-train_index, ]


table(train_data$Class)

################################################################################
## Oversample-SMOTE Technique used
data_oversample <- ovun.sample(Class ~ ., data = train_data, method = "over")$data
table(data_oversample$Class)#Check Balance: N:3143, Y:3124

################################################################################
##Chi-Square Attribute Selection Method

#Convert class to factor
data_oversample$Class <- as.factor(data_oversample$Class)

# Correct approach to apply the Chi-Square test and extract p-values directly
results <- sapply(
  data_oversample[, -which(names(data_oversample) == "Class")],
  function(x) {
    # Creating a contingency table
    tbl <- table(data_oversample$Class, x)
    
    # Check for low expected frequencies in any cell
    if(any(prop.table(tbl) * sum(tbl) < 5)) {
      p_value <- NA  # Assign NA if expected frequencies are too low
      } 
    else {
      # Performing the chi-squared test and accessing the p-value directly
      test_result <- chisq.test(tbl)
      p_value <- test_result$p.value
    }
    
    return(p_value)
  }
  )

# Optionally, filter based on a p-value threshold, e.g., p < 0.05
selected_features_chi_square <- names(results)[!is.na(results) & results < 0.05]
print(selected_features_chi_square)


#Over_sample training subset1 (Chi_square)
selected_features <- c(selected_features_chi_square, "Class")
Chi_Sq_1 <- data_oversample[, selected_features]
Chi_Sq_1_test <- test_data[, selected_features]


# Chi_Sq_1

#######################################################################################################

#Attribute Selection Method
## Anova F-test 
results_anova <- sapply(data_oversample[, sapply(data_oversample, is.numeric)], function(x) {
  # Conduct ANOVA
  model <- aov(x ~ data_oversample$Class)
  
  # Extract the summary and then the p-value
  p_value <- summary(model)[[1]]$`Pr(>F)`[1]
  
  return(p_value)
})

# Ensure results_anova is numeric for comparison
results_anova <- unlist(results_anova)

# Select features based on a p-value threshold p < 0.05
selected_features_anova <- names(results_anova)[results_anova < 0.05]

# Print the selected features
print(selected_features_anova)


#Over_sample training subset2(Anova)
selected_features_2 <- c(selected_features_anova, "Class")
Anov_F_1 <- data_oversample[, selected_features_2]
Anov_F_1_test <- test_data[, selected_features_2]


########################################################################################################
# Random Forest Attribute Selection Method
set.seed(123)
x <- data_oversample[, -which(names(data_oversample) == "Class")]  
y <- data_oversample$Class
y <- factor(y)


# Perform Random Forest Variable Importance
rf_model <- randomForest(x, y)
rf_selected_features <- importance(rf_model)
print(rf_selected_features)


# Extract the column corresponding to MeanDecreaseGini
mean_decrease_gini <- rf_selected_features[, "MeanDecreaseGini"]

# Sort the importance measures by decreasing order
sorted_importance <- sort(mean_decrease_gini, decreasing = TRUE)

# Select the top 100 features based on their importance
num_top_features <- 100  
selected_features_random_forest <- names(sorted_importance)[1:num_top_features]

# Print or use top_features
print(selected_features_random_forest)


#Over_sample training subset3 (Random Forest Variable Importance)
selected_features_3 <- c(selected_features_random_forest, "Class")
RF_1 <- data_oversample[, selected_features_3]
RF_1_test <- test_data[, selected_features_3]


#################################################################################################################

##UnderSample

data_N <- subset(train_data, Class == 0)
data_Y <- subset(train_data, Class == 1)

# Sample from the majority class ("N") to match the number of observations in the minority class ("Y")
set.seed(123) # for reproducibility
data_N_sampled <- data_N[sample(nrow(data_N), nrow(data_Y)), ]

# Combine the down-sampled majority class with the original minority class
balanced_data <- rbind(data_Y, data_N_sampled)

# Shuffle the balanced dataset (optional but recommended for random distribution)
set.seed(456) # for reproducibility
data_undersample <- balanced_data[sample(nrow(balanced_data)), ]

table(balanced_data$Class) # Check: Class undersample : N=686,Y=686


##########################################################################################
#UnderSample: Chi_Square
data_undersample$Class <- as.factor(data_undersample$Class)

# Correct approach to apply the Chi-Square test and extract p-values directly
results <- sapply(data_undersample[, -which(names(data_undersample) == "Class")], function(x) {
  # Creating a contingency table
  tbl <- table(data_undersample$Class, x)
  
  # Check for low expected frequencies in any cell
  if(any(prop.table(tbl) * sum(tbl) < 5)) {
    p_value <- NA  # Assign NA if expected frequencies are too low
  } else {
    # Performing the chi-squared test and accessing the p-value directly
    test_result <- chisq.test(tbl)
    p_value <- test_result$p.value
  }
  
  return(p_value)
})

# Optionally, filter based on a p-value threshold, e.g., p < 0.05
under_selected_features_chi_square <- names(results)[!is.na(results) & results < 0.05]
print(under_selected_features_chi_square)


#Under_sample training subset1 (Chi_square)
selected_features <- c(under_selected_features_chi_square, "Class")
Chi_Sq_2 <- data_undersample[, selected_features]
Chi_Sq_2_test <- test_data[, selected_features]



##########################################################################################
#UnderSample: Anova F-Test

results_anova <- sapply(data_undersample[, sapply(data_undersample, is.numeric)], function(x) {
  # Conduct ANOVA
  model <- aov(x ~ data_undersample$Class)
  
  # Extract the summary and then the p-value
  p_value <- summary(model)[[1]]$`Pr(>F)`[1]
  
  return(p_value)
})

# Ensure results_anova is numeric for comparison
results_anova <- unlist(results_anova)

# Select features based on a p-value threshold p < 0.05
under_selected_features_anova <- names(results_anova)[results_anova < 0.05]

# Print the selected features
print(under_selected_features_anova)


#Under_sample training subset2(Anova)
under_selected_features_2 <- c(under_selected_features_anova, "Class")
Anov_F_2 <- data_undersample[, selected_features_2]
Anov_F_2_test <- test_data[, selected_features_2]


########################################################################################################
# Random Forest Attribute Selection Method

set.seed(123)
x <- data_undersample[, -which(names(data_undersample) == "Class")]  
y <- data_undersample$Class
y <- factor(y)


# Perform Random Forest Variable Importance
rf_model <- randomForest(x, y)
under_rf_selected_features <- importance(rf_model)
print(under_rf_selected_features)


# Extract the column corresponding to MeanDecreaseGini
mean_decrease_gini <- under_rf_selected_features[, "MeanDecreaseGini"]

# Sort the importance measures by decreasing order
sorted_importance <- sort(mean_decrease_gini, decreasing = TRUE)

# Select the top 100 features based on their importance
num_top_features <- 100  
under_selected_features_random_forest <- names(sorted_importance)[1:num_top_features]

# Print or use top_features
print(under_selected_features_random_forest)


#Under_sample training subset3 (Random Forest Variable Importance)
under_selected_features_3 <- c(under_selected_features_random_forest, "Class")

under_RF_1 <- data_undersample[, under_selected_features_3]
under_RF_1_test <- test_data[, under_selected_features_3]


# View Random Forest Subset
head(under_RF_1)


################################################################################
## Model Prep ##


################################################################################
## Naive Bayes ##
#Best Laplace=0,usekernel = TRUE and adjust = 1.
perform_NB <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  test_data$Class <- factor(test_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  
  # training control
  trainControl <- trainControl(method = "cv", number = 10)
  
  # build a Naïve Bayes model from training dataset
  nb_classify <- train(Class ~ ., data = training_data, method = "naive_bayes",
        trControl = trainControl)
  
  # nb_classify <- naiveBayes(Class ~ ., data=training_data, laplace = 0, 
  # trControl=trainControl())
  # test on test dataset
  prob_pred <- predict(nb_classify, newdata = test_data, type = "prob")
  
  # performance measures on the test dataset
  pred <- factor(ifelse(prob_pred$Y >= 0.5, "Y", "N"))
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$mcc = mcc(pred, test_data$Class)
  results$model = c("NB")
  results$data_name = c(data_name)
  
  cm_out = as.table(cm)
  write.table(cm_out, sprintf("NB_%s_confusion_matrix.csv", data_name))
  
  ## ROC
  jpeg(sprintf("NB_%s_ROC.jpg", data_name), width = 350, height = 350)
  rocobj=roc(test_data$Class, prob_pred$Y, levels = c("N", "Y"), direction="<") 
  plot(rocobj, main ="NB ROC curve", 
       print.auc=TRUE)
  dev.off()
  
  return (results)
}


## Fine-Tuned J48 Decision Tree ##
perform_J48 <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  test_data$Class <- factor(test_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  
  # training_data$Class <- factor(training_data$Class)
  # test_data$Class <- factor(test_data$Class)
  
  # build Weka’s J48 decision tree model
  train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5,
                                summaryFunction = defaultSummary)
  J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
  DT_J48_model <- train(Class ~ ., data = training_data, method = "J48",
                        trControl = train_control, tuneGrid = J48Grid)
  
  
  #Best Decision Tree: Best C= 0.5, M=1
  OS_Best_J48 <-J48(Class ~ ., data = training_data, control = Weka_control(C = 0.5, M = 1))
  
  # test the model on the test dataset
  prob_pred <- predict(OS_Best_J48, newdata = test_data, type = "probability")
  
  prob_pred <- as.data.frame(prob_pred)
  
  # performance measures on the test dataset
  pred <- factor(ifelse(prob_pred$Y >= 0.5, "Y", "N"))
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$mcc = mcc(pred, test_data$Class)
  results$model = c("J48")
  results$data_name = c(data_name)
  
  cm_out = as.table(cm)
  write.table(cm_out, sprintf("J48_%s_confusion_matrix.csv", data_name))
  
  ## ROC
  jpeg(sprintf("J48_%s_ROC.jpg", data_name), width = 350, height = 350)
  rocobj=roc(test_data$Class, prob_pred$Y, levels = c("N", "Y"), direction="<") 
  plot(rocobj, main ="J48 ROC curve", 
       print.auc=TRUE)
  dev.off()
  
  return (results)
  }

## Basic C5.0 Decision Tree ##
perform_C5 <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  test_data$Class <- factor(test_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  
  C5_tree <- C5.0(Class ~ ., data = training_data)
  
  # test 
  prob_pred <- predict(C5_tree, newdata = test_data, type = "prob")
  
  prob_pred <- as.data.frame(prob_pred)
  
  # performance measures on the test dataset
  pred <- factor(ifelse(prob_pred$Y >= 0.5, "Y", "N"))
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$mcc = mcc(pred, test_data$Class)
  results$model = c("C5")
  results$data_name = c(data_name)
  
  cm_out = as.table(cm)
  write.table(cm_out, sprintf("C5_%s_confusion_matrix.csv", data_name))
  
  ## ROC
  jpeg(sprintf("C5_%s_ROC.jpg", data_name), width = 350, height = 350)
  rocobj=roc(test_data$Class, prob_pred$Y, levels = c("N", "Y"), direction="<") 
  plot(rocobj, main ="C5 ROC curve", 
       print.auc=TRUE)
  dev.off()
  
  return (results)
  }

## Basic Default Logistic Regression ##
perform_LogR <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  test_data$Class <- factor(test_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  
  # set up binomial logisitic regression model
  logitModel <- glm(Class ~ ., data = training_data, family = "binomial") 
  
  # use predict() with type = "response" to compute predicted probabilities. 
  logitModel.pred <- predict(logitModel, test_data, type = "response")
  
  # performance measures on the test dataset
  pred <- factor(ifelse(logitModel.pred >= 0.5, "Y", "N"))
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$mcc = mcc(pred, test_data$Class)
  results$model = c("LogReg")
  results$data_name = c(data_name)
  
  cm_out = as.table(cm)
  write.table(cm_out, sprintf("LogR_%s_confusion_matrix.csv", data_name))
  
  ## ROC
  jpeg(sprintf("LogR_%s_ROC.jpg", data_name), width = 350, height = 350)
  rocobj=roc(test_data$Class, logitModel.pred, levels = c("N", "Y"), direction="<") 
  plot(rocobj, main ="LogR ROC curve", 
       print.auc=TRUE)
  dev.off()
  
  
  return (results)
  }


## Custom GLMNet Logistic Regression ##
perform_glmNet_LogR <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  test_data$Class <- factor(test_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  
  # rename for simplicity
  x <- model.matrix(Class ~ ., data = training_data)[,-1] # Removing intercept
  y <- training_data$Class
  
  # Define control using caret's trainControl for cross-validation
  train_control <- trainControl(method = "cv", 
                                number = 5,
                                classProbs = TRUE, 
                                summaryFunction = twoClassSummary)
  alpha <- 0.5
  
  # Set up a grid of lambda (regularization parameter) values
  lambda_grid <- 10^seq(-3, 3, length = 100)
  
  
  # Fit model with cv.glmnet
  set.seed(123)
  cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = alpha, lambda = lambda_grid) 
  
  # Best lambda
  best_lambda <- cv_fit$lambda.min
  set.seed(123)
  caret_model <- train(x, y, 
                       method = "glmnet",
                       trControl = train_control,
                       tuneGrid = expand.grid(alpha = alpha, lambda = lambda_grid),
                       metric = "ROC")
  
  # use predict() with type = "prob" to compute predicted probabilities. 
  prob_pred <- predict(caret_model, test_data, type = "prob")
  
  # performance measures on the test dataset
  pred <- factor(ifelse(prob_pred$Y >= 0.5, "Y", "N"))

  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$mcc = mcc(pred, test_data$Class)
  results$model = c("Custom_LogReg")
  results$data_name = c(data_name)
  
  cm_out = as.table(cm)
  write.table(cm_out, sprintf("GLM_LogR_%s_confusion_matrix.csv", data_name))
  
  ## ROC
  jpeg(sprintf("GLM_LogR_%s_ROC.jpg", data_name), width = 350, height = 350)
  rocobj=roc(test_data$Class, prob_pred$Y, levels = c("N", "Y"), direction="<") 
  plot(rocobj, main ="GLM LogR ROC curve", 
       print.auc=TRUE)
  dev.off()
  
  return (results)
}


## SVM ##
perform_SVM <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  test_data$Class <- factor(test_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  
  features <- training_data[, -which(names(training_data) == "Class")]
  target <- training_data$Class
  gamma_value <- 1 / (2 * 0.4^2)
  c_val <- 1 # needed to use a small c
  
  OS_Best_svm_model <- ksvm(x = as.matrix(features), 
                            y = as.factor(target), 
                            type = "C-svc", 
                            kernel = "rbfdot", 
                            C = c_val, 
                            kpar = list(sigma = 0.1), 
                            prob.model = TRUE)
  
  testing_features = test_data[, -which(names(test_data) == "Class")]
  # test the model on the test dataset
  prob_pred <- predict(OS_Best_svm_model,
                  newdata = testing_features,
                  type = "prob")
  
  prob_pred <- as.data.frame(prob_pred)
  
  # performance measures on the test dataset
  pred <- factor(ifelse(prob_pred$Y >= 0.5, "Y", "N"))
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$mcc = mcc(pred, test_data$Class)
  results$model = c("SVM")
  results$data_name = c(data_name)
  
  cm_out = as.table(cm)
  write.table(cm_out, sprintf("SVM_%s_confusion_matrix.csv", data_name))
  
  ## ROC
  jpeg(sprintf("SVM_%s_ROC.jpg", data_name), width = 350, height = 350)
  rocobj=roc(test_data$Class, prob_pred$Y, levels = c("N", "Y"), direction="<") 
  plot(rocobj, main ="SVM LogR ROC curve", 
       print.auc=TRUE)
  dev.off()
  
  return (results)
}


## RF ##
perform_RF <- function(training_data, test_data, data_name){
  print("calculating RF")
  features <- training_data[, -which(names(training_data) == "Class")]
  test_features <- test_data[, -which(names(test_data) == "Class")]
  training_data$Class <- factor(training_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  test_data$Class <- factor(test_data$Class, levels = c("0", "1"), labels = c("N", "Y"))
  
  ctrl <- trainControl(method = "CV",
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE, savePredictions = TRUE)

  rfFit <- train(x = features, y = training_data$Class, method = "rf",
                 ntree = 100,
                 # tuneGrid = data.frame(mtry = mtryValues),
                 importance = TRUE,
                 metric = "ROC",
                 trControl = ctrl
                 )
  
  # test the model on the test dataset
  prob_pred <- predict(rfFit,
                  newdata = test_features,
                  type = "prob")
  
  # performance measures on the test dataset
  pred <- factor(ifelse(prob_pred$Y >= 0.5, "Y", "N"))

    # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$mcc = mcc(pred, test_data$Class)
  results$model = c("RF")
  results$data_name = c(data_name)
  
  cm_out = as.table(cm)
  write.table(cm_out, sprintf("RF_%s_confusion_matrix.csv", data_name))
  
  ## ROC
  jpeg(sprintf("RF_%s_ROC.jpg", data_name), width = 350, height = 350)
  rocobj=roc(test_data$Class, prob_pred$Y, levels = c("N", "Y"), direction="<") 
  plot(rocobj, main ="RF LogR ROC curve", 
       print.auc=TRUE)
  dev.off()
  
  return (results)
}

################################################################################
### Train and Evaluate Models ###


## Over Sample Set
oversample_set = list(Chi_Sq_1, Anov_F_1, RF_1)
oversample_test_set = list(Chi_Sq_1_test, Anov_F_1_test, RF_1_test)
oversample_set_names = c(
  "chi_square_oversample",
  "anova_oversample",
  "RF_oversample"
  )

# let's get a baseline with of the entire train set with no sampling
base_LogR_results = perform_LogR(train_data, test_data, "all_data")
over_sample_output_df = base_LogR_results


for (i in 1:length(oversample_set)){
  J48_results_o = perform_J48(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])

  NB_results_o = perform_NB(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])

  C5_results_o = perform_C5(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])
  
  RF_results_o = perform_RF(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])

  LogR_results_o = perform_LogR(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])
  
  optimze_LogR_results_o = perform_glmNet_LogR(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])

  svm_results_o = perform_SVM(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])
  
  over_sample_output_df = rbind(over_sample_output_df,
                                NB_results_o,
                                J48_results_o,
                                C5_results_o,
                                LogR_results_o,
                                optimze_LogR_results_o,
                                RF_results_o,
                                svm_results_o)
  
}

over_sample_output_df <- apply(over_sample_output_df,2,as.character)
write.csv(over_sample_output_df, "over_sample_results.csv")


### UNDER SAMPLE SET
undersample_set = list(Chi_Sq_2, Anov_F_2, under_RF_1)
undersample_test_set = list(Chi_Sq_2_test, Anov_F_2_test, under_RF_1_test)

undersample_set_names = list(
  "chi_square_undersample",
  "anova_undersample",
  "RF_undersample"
)

# let's get a baseline with of the entire train set with no sampling
base_LogR_results = perform_LogR(train_data, test_data, "all_data")
under_sample_output_df = base_LogR_results


for (i in 1:length(undersample_set)){
  J48_results = perform_J48(undersample_set[[i]], undersample_test_set[[i]], undersample_set_names[i])
  
  NB_results = perform_NB(undersample_set[[i]], undersample_test_set[[i]], undersample_set_names[i])
  
  C5_results = perform_C5(undersample_set[[i]], undersample_test_set[[i]], undersample_set_names[i])
  
  RF_results = perform_RF(undersample_set[[i]], undersample_test_set[[i]], undersample_set_names[i])
  
  ogR_results = perform_LogR(undersample_set[[i]], undersample_test_set[[i]], undersample_set_names[i])
  
  optimze_LogR_results = perform_glmNet_LogR(undersample_set[[i]], undersample_test_set[[i]], undersample_set_names[i])

  svm_results = perform_SVM(undersample_set[[i]], undersample_test_set[[i]], undersample_set_names[i])
  
  under_sample_output_df = rbind(under_sample_output_df,
                                 NB_results,
                                 J48_results,
                                 C5_results,
                                 LogR_results,
                                 optimze_LogR_results,
                                 RF_results,
                                 svm_results)
  
}

# save results
under_sample_output_df <- apply(under_sample_output_df,2,as.character)
write.csv(under_sample_output_df, "under_sample_results.csv")


# test_results = perform_NB(Anov_F_2, Anov_F_2_test, undersample_set_names[i])

