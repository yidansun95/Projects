library(caret)
library(ROSE)
library(dplyr)
library(stats)
library(randomForest)
library(e1071) #Decision Tree
library(caret) #Confusion matrix
library(nnet)
library(RWeka)

library(glmnet)

#SVM
library(kernlab)


library(FSelector)
library(Boruta)
library(rsample)
library(tidyverse)
library(C50)
library(rpart)
library(rpart.plot)
library(pROC)
library(MASS)
library(mltools)
library(psych)



df <- read.csv(file = "E:\\BU\\2024spring\\CS 699\\project\\Preprocessed_data_Sun, Yidan+Brian Ramm\\cleaned_trimmed_data.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Class: N for 3926, Y for 925, we need to either Oversample of Y or Undersample N
table(df$Class)
set.seed(123)

# Split the dataset into training and testing sets
train_index <- createDataPartition(y = df$Class, p = 0.8, list = FALSE)

# Create training and testing sets
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

table(train_data$Class)
#############################################################################################
## Oversample-SMOTE Technique used
library(ROSE)
data_oversample <- ovun.sample(Class ~ ., data = train_data, method = "over")$data
table(data_oversample$Class)#Check Balance: N:3143, Y:3124

#############################################################################################
##Chi-Square Attribute Selection Method
library(dplyr)
#Convert class to factor
data_oversample$Class <- as.factor(data_oversample$Class)

# Correct approach to apply the Chi-Square test and extract p-values directly
results <- sapply(data_oversample[, -which(names(data_oversample) == "Class")], function(x) {
  # Creating a contingency table
  tbl <- table(data_oversample$Class, x)
  
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

print(results)

# Filter based on a p-value threshold, e.g., p < 0.05
selected_features_chi_square <- names(results)[!is.na(results) & results < 0.05]
print(selected_features_chi_square)


#Over_sample training subset1 (Chi_square)
selected_features <- c(selected_features_chi_square, "Class")
Chi_Sq_1 <- data_oversample[, selected_features]
Chi_Sq_1_test <- test_data[, selected_features]

# View Chi_square training subset num
nrow(Chi_Sq_1)


#######################################################################################################

#Attribute Selection Method
## Anova F-test 
library(stats)

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



# View Anova F_test subset
head(Anov_F_1)

nrow(Anov_F_1)

########################################################################################################
# Random Forest Attribute Selection Method
library(randomForest)
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


# View Random Forest Subset
head(RF_1)

nrow(RF_1)


#################################################################################################################

##UnderSample
data_N <- subset(train_data, Class == 0) #No(0):738 Yes(1):738
data_Y <- subset(train_data, Class == 1)

# Check if subsets are non-empty
print(nrow(data_N))
print(nrow(data_Y))

set.seed(123)
data_N_sampled <- data_N[sample(nrow(data_N), nrow(data_Y)), ]

balanced_data <- rbind(data_Y, data_N_sampled)
data_undersample <- balanced_data[sample(nrow(balanced_data)), ]

table(data_undersample$Class)


##########################################################################################
#UnderSample: Chi_Square


##Chi-Square Attribute Selection Method
library(dplyr)
#Convert class to factor
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

print(results)

# Filter based on a p-value threshold, e.g., p < 0.05
selected_features_chi_square <- names(results)[!is.na(results) & results < 0.05]
print(selected_features_chi_square)


#Over_sample training subset1 (Chi_square)
selected_features <- c(selected_features_chi_square, "Class")
Chi_Sq_2 <- data_undersample[, selected_features]
Chi_Sq_2_test <- test_data[, selected_features]



# View Chi_square training subset num
nrow(Chi_Sq_2)




##########################################################################################
#UnderSample: Anova F-Test
library(stats)

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
Anov_F_2 <- data_undersample[, under_selected_features_2]
Anov_F_2_test <- test_data[, under_selected_features_2]



# View Anova F_test subset
head(Anov_F_2)

########################################################################################################
# Random Forest Attribute Selection Method
library(randomForest)
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
RF_2 <- data_undersample[, under_selected_features_3]
RF_2_test <- test_data[, under_selected_features_3]



# View Random Forest Subset
head(RF_2)


##################################################################################################################
#Over_Sample Models Training
#################################################################################################################
#Knn--Anov_F_1
set.seed(123)
levels(Anov_F_1$Class)
Anov_F_1$Class <- make.names(Anov_F_1$Class)


train_control <- trainControl(
  method = "cv",        # Specify cross-validation
  number = 10,          # Number of folds in cross-validation
  savePredictions = "final",  # Option to save predictions
  classProbs = TRUE     # Whether to save class probabilities
)

##KNN best k=5
knnModel <- train(Class ~., data = Anov_F_1,
                  method = "knn", trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 10)
knnModel

##Best Knn:k=5
OS_Best_Knn_Anov_F_1 <- knn3(Class ~ ., data = Anov_F_1, k = 5, prob = TRUE)





#Knn--Chi_Sq_1
set.seed(123)

levels(Chi_Sq_1$Class)
Chi_Sq_1$Class <- make.names(Chi_Sq_1$Class)

train_control <- trainControl(
  method = "cv",        # Specify cross-validation
  number = 10,          # Number of folds in cross-validation
  savePredictions = "final",  # Option to save predictions
  classProbs = TRUE     # Whether to save class probabilities
)

knnModel <- train(Class ~., data = Chi_Sq_1,
                  method = "knn", trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 10)
knnModel

##Best Knn:k=5
OS_Best_Knn_Chi_Sq_1 <- knn3(Class ~ ., data = Chi_Sq_1, k = 5, prob = TRUE)






#Knn--RF_1
set.seed(123)

levels(RF_1$Class)
RF_1$Class <- make.names(RF_1$Class)

train_control <- trainControl(
  method = "cv",        # Specify cross-validation
  number = 10,          # Number of folds in cross-validation
  savePredictions = "final",  # Option to save predictions
  classProbs = TRUE     # Whether to save class probabilities
)

knnModel <- train(Class ~., data = RF_1,
                  method = "knn", trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 10)
knnModel

##Best Knn:k=5
OS_Best_Knn_RF_1 <- knn3(Class ~ ., data = RF_1, k = 5, prob = TRUE)


####################################################################################################
##Decision Tree rpart
#DecisionTree--Anov_F_1
DT_model <- train(Class ~ ., data = Anov_F_1, method = "rpart",
               trControl = train_control, tuneLength = 10)
DT_model

#Best Decision Tree: Best cp is 0.004321383
OS_Best_DecisionTree_Anov_F_1 <- train(Class ~ ., data = Anov_F_1, method = "rpart",
                            trControl = train_control,
                            tuneGrid = data.frame(cp = 0.004321383))




#DecisionTree--Chi_Sq_1
DT_model <- train(Class ~ ., data = Chi_Sq_1, method = "rpart",
                  trControl = train_control, tuneLength = 10)
DT_model

#Best Decision Tree: Best cp is 0.004321383
OS_Best_DecisionTree_Chi_Sq_1 <- train(Class ~ ., data = Chi_Sq_1, method = "rpart",
                                       trControl = train_control,
                                       tuneGrid = data.frame(cp = 0.004321383))



#DecisionTree--RF_1
DT_model <- train(Class ~ ., data = RF_1, method = "rpart",
                  trControl = train_control, tuneLength = 10)
DT_model

#Best Decision Tree: Best cp is 0.004321383
OS_Best_DecisionTree_RF_1 <- train(Class ~ ., data = RF_1, method = "rpart",
                                       trControl = train_control,
                                       tuneGrid = data.frame(cp = 0.003841229))


#####################################################################################################
##Decision Tree-J48
#J48--Anov_F_1
Anov_F_1$Class <- factor(Anov_F_1$Class)

set.seed(123)

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              summaryFunction = defaultSummary)
J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
DT_J48_model <- train(Class ~ ., data = Anov_F_1, method = "J48",
               trControl = train_control, tuneGrid = J48Grid)

DT_J48_model


#Best Decision Tree: Best C= 0.5, M=1
OS_Best_J48_Anov_F_1 <-J48(Class ~ ., data = Anov_F_1, control = Weka_control(C = 0.5, M = 1))




##J48--Chi_Sq_1
Chi_Sq_1$Class <- factor(Chi_Sq_1$Class)

set.seed(123)

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              summaryFunction = defaultSummary)
J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))


DT_J48_model <- train(Class ~ ., data = Chi_Sq_1, method = "J48",
                      trControl = train_control, tuneGrid = J48Grid)

DT_J48_model


#Best Decision Tree: Best C= 0.5, M=1
OS_Best_J48_Chi_Sq_1 <-J48(Class ~ ., data = Chi_Sq_1, control = Weka_control(C = 0.5, M = 1))





##J48--RF_1

RF_1$Class <- factor(RF_1$Class)

set.seed(123)

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              summaryFunction = defaultSummary)
J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))


DT_J48_model <- train(Class ~ ., data = RF_1, method = "J48",
                      trControl = train_control, tuneGrid = J48Grid)

DT_J48_model


#Best Decision Tree: Best C= 0.5, M=1
OS_Best_J48_RF_1 <-J48(Class ~ ., data = RF_1, control = Weka_control(C = 0.5, M = 1))




#############################################################################################################
## Logistics Regression

#Logistics Regression--Anov_F_1
x <- model.matrix(Class ~ ., data = Anov_F_1)[,-1] # Removing intercept
y <- Anov_F_1$Class

# Define control using caret's trainControl for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
alpha <- 0.5

# Set up a grid of lambda (regularization parameter) values
lambda_grid <- 10^seq(-3, 3, length = 100)

# Fit model with cv.glmnet
set.seed(123)
cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = alpha, lambda = lambda_grid)

# Best lambda
best_lambda <- cv_fit$lambda.min
set.seed(123)
caret_model <- train(x, y, method = "glmnet",
                     trControl = train_control,
                     tuneGrid = expand.grid(alpha = alpha, lambda = lambda_grid),
                     metric = "ROC")

# Print best model details
print(caret_model)


#Best  Logistic Model: alpha = 0.5 and lambda = 0.001519911.
OS_Best_LogR_Anov_F_1 <- glmnet(x, y, family = "binomial", alpha = 0.5, lambda = 0.001519911)




#Logistics Regression--Chi_Sq_1

x <- model.matrix(Class ~ ., data = Chi_Sq_1)[,-1] # Removing intercept
y <- Chi_Sq_1$Class

# Define control using caret's trainControl for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
alpha <- 0.5

# Set up a grid of lambda (regularization parameter) values
lambda_grid <- 10^seq(-3, 3, length = 100)

# Fit model with cv.glmnet
set.seed(123)
cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = alpha, lambda = lambda_grid)

# Best lambda
best_lambda <- cv_fit$lambda.min
set.seed(123)
caret_model <- train(x, y, method = "glmnet",
                     trControl = train_control,
                     tuneGrid = expand.grid(alpha = alpha, lambda = lambda_grid),
                     metric = "ROC")

# Print best model details
print(caret_model)


#Best  Logistic Model: alpha = 0.5 and lambda = 0.002656088
OS_Best_LogR_Chi_Sq_1 <- glmnet(x, y, family = "binomial", alpha = 0.5, lambda = 0.002656088)




#Logistics Regression--RF_1

x <- model.matrix(Class ~ ., data = RF_1)[,-1] # Removing intercept
y <- RF_1$Class

# Define control using caret's trainControl for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
alpha <- 0.5

# Set up a grid of lambda (regularization parameter) values
lambda_grid <- 10^seq(-3, 3, length = 100)

# Fit model with cv.glmnet
set.seed(123)
cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = alpha, lambda = lambda_grid)

# Best lambda
best_lambda <- cv_fit$lambda.min
set.seed(123)
caret_model <- train(x, y, method = "glmnet",
                     trControl = train_control,
                     tuneGrid = expand.grid(alpha = alpha, lambda = lambda_grid),
                     metric = "ROC")

# Print best model details
print(caret_model)


#Best  Logistic Model: alpha = 0.5 and lambda = 0.002656088
OS_Best_LogR_RF_1 <- glmnet(x, y, family = "binomial", alpha = 0.5, lambda = 0.002656088)


##########################################################################################################

#Naive Bayes

#Naive Bayes--Anov_F_1
library(e1071)
set.seed(123)
trainControl <- trainControl(method = "cv", number = 10)

NB_model <- train(Class ~ ., data = Anov_F_1, method = "naive_bayes",
               trControl = trainControl)
NB_model

#Best Laplace=0,usekernel = FALSE and adjust = 1.
OS_Best_NB_Anov_F_1 <- naiveBayes(x, y, laplace = 0, usekernel = FALSE, adjust = 1)




#Naive Bayes--Chi_Sq_1
library(e1071)
set.seed(123)
trainControl <- trainControl(method = "cv", number = 10)

NB_model <- train(Class ~ ., data = Chi_Sq_1, method = "naive_bayes",
                  trControl = trainControl)
NB_model



#Best Laplace=0,usekernel = FALSE and adjust = 1.
OS_Best_NB_Chi_Sq_1 <- naiveBayes(x, y, laplace = 0, usekernel = FALSE, adjust = 1)



#Naive Bayes--RF_1
library(e1071)
set.seed(123)
trainControl <- trainControl(method = "cv", number = 10)

NB_model <- train(Class ~ ., data = RF_1, method = "naive_bayes",
                  trControl = trainControl)
NB_model

#Best Laplace=0,usekernel = FALSE and adjust = 1.
OS_Best_NB_RF_1 <- naiveBayes(x, y, laplace = 0, usekernel = FALSE, adjust = 1)

###################################################################################################
#SVM
#SVM--Anov_F_1
set.seed(123)
subset_size <- 100

sampled_indices <- sample(nrow(Anov_F_1), size = subset_size)
subset_dataset <- Anov_F_1[sampled_indices, ]

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

SVM_model <- caret::train(Class ~ ., data = subset_dataset, method = "svmRadial",
                      preProc = c("center", "scale"),
                      trControl = train_control, tuneGrid = svmGrid)
SVM_model


features <- subset_dataset[, -which(names(subset_dataset) == "Class")]
target <- subset_dataset$Class
gamma_value <- 1 / (2 * 0.4^2)


#Best Value sigma = 0.4 and C = 1:
OS_Best_SVM_Anov_F_1 <- ksvm(x = as.matrix(features), 
                        y = as.factor(target), 
                        type = "C-svc", 
                        kernel = "rbfdot", 
                        C = 1.0, 
                        kpar = list(sigma = 0.4), 
                        prob.model = TRUE)




#SVM--Chi_Sq_1
set.seed(123)
subset_size <- 100

sampled_indices <- sample(nrow(Chi_Sq_1), size = subset_size)
subset_dataset <- Chi_Sq_1[sampled_indices, ]

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

SVM_model <- caret::train(Class ~ ., data = subset_dataset, method = "svmRadial",
                          preProc = c("center", "scale"),
                          trControl = train_control, tuneGrid = svmGrid)
SVM_model


features <- subset_dataset[, -which(names(subset_dataset) == "Class")]
target <- subset_dataset$Class
gamma_value <- 1 / (2 * 0.4^2)


#Best Value sigma = 0.4 and C = 1:
OS_Best_SVM_Chi_Sq_1 <- ksvm(x = as.matrix(features), 
                             y = as.factor(target), 
                             type = "C-svc", 
                             kernel = "rbfdot", 
                             C = 1.0, 
                             kpar = list(sigma = 0.4), 
                             prob.model = TRUE)





#SVM--RF_1
set.seed(123)
subset_size <- 100

sampled_indices <- sample(nrow(RF_1), size = subset_size)
subset_dataset <- RF_1[sampled_indices, ]

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

SVM_model <- caret::train(Class ~ ., data = subset_dataset, method = "svmRadial",
                          preProc = c("center", "scale"),
                          trControl = train_control, tuneGrid = svmGrid)
SVM_model


features <- subset_dataset[, -which(names(subset_dataset) == "Class")]
target <- subset_dataset$Class
gamma_value <- 1 / (2 * 0.4^2)


#Best Value sigma = 0.4 and C = 1:
OS_Best_SVM_RF_1 <- ksvm(x = as.matrix(features), 
                             y = as.factor(target), 
                             type = "C-svc", 
                             kernel = "rbfdot", 
                             C = 1.0, 
                             kpar = list(sigma = 0.4), 
                             prob.model = TRUE)




################################################################################################################
##Random Forest
#Random Forest--Anov_F_1

set.seed(123)
subset_size <- 300
sampled_indices <- sample(nrow(Anov_F_1), size = subset_size)
subset <- Anov_F_1[sampled_indices, ]
subset$Class <- factor(subset$Class)
clean_levels <- make.names(levels(subset$Class), unique = TRUE)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(subset)-1, by = 1)




# Reassign cleaned levels back to the factor
levels(subset$Class) <- clean_levels

# Model Training
rfFit <- train(x = subset[,-13], y = subset$Class, method = "rf",
               ntree = 500, tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE, metric = "ROC", trControl = ctrl)
rfFit




# Instantiate the Random Forest best model with mtry = 5
OS_Best_RF_Anov_F_1 <- randomForest(Class ~ ., data = subset, mtry = 5, ntree = 500)





#Random Forest--Chi_Sq_1

set.seed(123)
subset_size <- 300
sampled_indices <- sample(nrow(Chi_Sq_1), size = subset_size)
subset <- Chi_Sq_1[sampled_indices, ]
subset$Class <- factor(subset$Class)
clean_levels <- make.names(levels(subset$Class), unique = TRUE)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(subset)-1, by = 1)




# Reassign cleaned levels back to the factor
levels(subset$Class) <- clean_levels

# Model Training
rfFit <- train(x = subset[,-13], y = subset$Class, method = "rf",
               ntree = 500, tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE, metric = "ROC", trControl = ctrl)
rfFit




# Instantiate the Random Forest best model with mtry = 3
OS_Best_RF_Chi_Sq_1 <- randomForest(Class ~ ., data = subset, mtry = 3, ntree = 500)








#Random Forest--RF_1

set.seed(123)
subset_size <- 300
sampled_indices <- sample(nrow(RF_1), size = subset_size)
subset <- RF_1[sampled_indices, ]
subset$Class <- factor(subset$Class)
clean_levels <- make.names(levels(subset$Class), unique = TRUE)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(subset)-1, by = 1)




# Reassign cleaned levels back to the factor
levels(subset$Class) <- clean_levels

# Model Training
rfFit <- train(x = subset[,-13], y = subset$Class, method = "rf",
               ntree = 500, tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE, metric = "ROC", trControl = ctrl)
rfFit




# Instantiate the Random Forest best model with mtry = 5
OS_Best_RF_RF_1 <- randomForest(Class ~ ., data = subset, mtry = 5, ntree = 500)





##########################################################################################################
#Under_Sample
###########################################################################################################

#Knn
levels(Anov_F_2$Class)
Anov_F_2$Class <- make.names(Anov_F_2$Class)

set.seed(123)
train_control <- trainControl(
  method = "cv",        # Specify cross-validation
  number = 10,          # Number of folds in cross-validation
  savePredictions = "final",  # Option to save predictions
  classProbs = TRUE     # Whether to save class probabilities
)

knnModel <- train(Class ~., data = Anov_F_2,
                  method = "knn", trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 10)
knnModel

##Best Knn:k=15
US_Best_Knn_Anov_F_2 <- knn3(Class ~ ., data = Anov_F_2, k = 15, prob = TRUE)





#Knn--Chi_Sq_2
levels(Chi_Sq_2$Class)
Chi_Sq_2$Class <- make.names(Chi_Sq_2$Class)

set.seed(123)
train_control <- trainControl(
  method = "cv",        # Specify cross-validation
  number = 10,          # Number of folds in cross-validation
  savePredictions = "final",  # Option to save predictions
  classProbs = TRUE     # Whether to save class probabilities
)


knnModel <- train(Class ~., data = Chi_Sq_2,
                  method = "knn", trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 10)
knnModel

##Best Knn:k=11
US_Best_Knn_Chi_Sq_2 <- knn3(Class ~ ., data = Chi_Sq_2, k = 11, prob = TRUE)






#Knn--RF_2
levels(RF_2$Class)
RF_2$Class <- make.names(RF_2$Class)



set.seed(123)
train_control <- trainControl(
  method = "cv",        # Specify cross-validation
  number = 10,          # Number of folds in cross-validation
  savePredictions = "final",  # Option to save predictions
  classProbs = TRUE     # Whether to save class probabilities
)

knnModel <- train(Class ~., data = RF_2,
                  method = "knn", trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 10)
knnModel

##Best Knn:k=23
US_Best_Knn_RF_2 <- knn3(Class ~ ., data = RF_2, k = 23, prob = TRUE)


##############################################################################################
##Decision Tree rpart
#DecisionTree--Anov_F_2
DT_model <- train(Class ~ ., data = Anov_F_2, method = "rpart",
                  trControl = train_control, tuneLength = 10)
DT_model

#Best Decision Tree: Best cp is 0.004742547
US_Best_DecisionTree_Anov_F_2 <- train(Class ~ ., data = Anov_F_2, method = "rpart",
                                       trControl = train_control,
                                       tuneGrid = data.frame(cp = 0.004742547))




#DecisionTree--Chi_Sq_2
DT_model <- train(Class ~ ., data = Chi_Sq_2, method = "rpart",
                  trControl = train_control, tuneLength = 10)
DT_model

#Best Decision Tree: Best cp is 0.003387534
US_Best_DecisionTree_Chi_Sq_2 <- train(Class ~ ., data = Chi_Sq_2, method = "rpart",
                                       trControl = train_control,
                                       tuneGrid = data.frame(cp = 0.003387534))



#DecisionTree--RF_2
DT_model <- train(Class ~ ., data = RF_2, method = "rpart",
                  trControl = train_control, tuneLength = 10)
DT_model

#Best Decision Tree: Best cp is 0.006775068
US_Best_DecisionTree_RF_2 <- train(Class ~ ., data = RF_2, method = "rpart",
                                   trControl = train_control,
                                   tuneGrid = data.frame(cp = 0.006775068))






########################################################################################################
##Decision Tree J48

#Decision Tree J48--Anov_F_2
Anov_F_2$Class <- factor(Anov_F_2$Class)

set.seed(123)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              summaryFunction = defaultSummary)
J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
DT_J48_model <- train(Class ~ ., data = Anov_F_2, method = "J48",
                      trControl = train_control, tuneGrid = J48Grid)

DT_J48_model

#Best Decision Tree: Best C=0.01, M=4
US_Best_J48_Anov_F_2  <- J48(Class ~ ., data = Anov_F_2, control = Weka_control(C = 0.01, M = 4))


#Decision Tree J48--Chi_Sq_2

Chi_Sq_2$Class <- factor(Chi_Sq_2$Class)

set.seed(123)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              summaryFunction = defaultSummary)
J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
DT_J48_model <- train(Class ~ ., data = Chi_Sq_2, method = "J48",
                      trControl = train_control, tuneGrid = J48Grid)

DT_J48_model


#Best Decision Tree: Best C=0.01, M=1
US_Best_J48_Chi_Sq_2  <- J48(Class ~ ., data = Chi_Sq_2, control = Weka_control(C = 0.01, M = 1))




#Decision Tree J48--RF_2
RF_2$Class <- factor(RF_2$Class)

set.seed(123)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              summaryFunction = defaultSummary)
J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))

DT_J48_model <- train(Class ~ ., data = RF_2, method = "J48",
                      trControl = train_control, tuneGrid = J48Grid)

DT_J48_model


#Best Decision Tree: Best C=0.01, M=2
US_Best_J48_RF_2  <- J48(Class ~ ., data = RF_2, control = Weka_control(C = 0.01, M = 2))




#######################################################################################################
## Logistics Regression

#Logistics Regression--Anov_F_2
x <- model.matrix(Class ~ ., data = Anov_F_2)[,-1] # Removing intercept
y <- Anov_F_2$Class

# Define control using caret's trainControl for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
alpha <- 0.5

# Set up a grid of lambda (regularization parameter) values
lambda_grid <- 10^seq(-3, 3, length = 100)

# Fit model with cv.glmnet
set.seed(123)
cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = alpha, lambda = lambda_grid)

# Best lambda
best_lambda <- cv_fit$lambda.min
set.seed(123)
caret_model <- train(x, y, method = "glmnet",
                     trControl = train_control,
                     tuneGrid = expand.grid(alpha = alpha, lambda = lambda_grid),
                     metric = "ROC")

# Print best model details
print(caret_model)


#Best  Logistic Model: alpha = 0.5 and lambda = 0.01072267
US_Best_LogR_Anov_F_2 <- glmnet(x, y, family = "binomial", alpha = 0.5, lambda = 0.01072267)






#Logistics Regression--Chi_Sq_2

x <- model.matrix(Class ~ ., data = Chi_Sq_2)[,-1] # Removing intercept
y <- Chi_Sq_2$Class

# Define control using caret's trainControl for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
alpha <- 0.5

# Set up a grid of lambda (regularization parameter) values
lambda_grid <- 10^seq(-3, 3, length = 100)

# Fit model with cv.glmnet
set.seed(123)
cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = alpha, lambda = lambda_grid)

# Best lambda
best_lambda <- cv_fit$lambda.min
set.seed(123)
caret_model <- train(x, y, method = "glmnet",
                     trControl = train_control,
                     tuneGrid = expand.grid(alpha = alpha, lambda = lambda_grid),
                     metric = "ROC")

# Print best model details
print(caret_model)


#Best  Logistic Model: alpha = 0.5 and lambda = 0.00231013
US_Best_LogR_Chi_Sq_2 <- glmnet(x, y, family = "binomial", alpha = 0.5, lambda = 0.00231013)




#Logistics Regression--RF_2

x <- model.matrix(Class ~ ., data = RF_2)[,-1] # Removing intercept
y <- RF_2$Class

# Define control using caret's trainControl for cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
alpha <- 0.5

# Set up a grid of lambda (regularization parameter) values
lambda_grid <- 10^seq(-3, 3, length = 100)

# Fit model with cv.glmnet
set.seed(123)
cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = alpha, lambda = lambda_grid)

# Best lambda
best_lambda <- cv_fit$lambda.min
set.seed(123)
caret_model <- train(x, y, method = "glmnet",
                     trControl = train_control,
                     tuneGrid = expand.grid(alpha = alpha, lambda = lambda_grid),
                     metric = "ROC")

# Print best model details
print(caret_model)


#Best  Logistic Model: alpha = 0.5 and lambda = 0.002656088
US_Best_LogR_RF_2 <- glmnet(x, y, family = "binomial", alpha = 0.5, lambda = 0.02848036)



##################################################################################################
#Naive Bayes

#Naive Bayes--Anov_F_2
set.seed(123)
trainControl <- trainControl(method = "cv", number = 10)

NB_model <- train(Class ~ ., data = Anov_F_2, method = "naive_bayes",
                  trControl = trainControl)
NB_model

#Best Laplace=0,usekernel = FALSE and adjust = 1.
US_Best_NB_Anov_F_2 <- naiveBayes(x, y, laplace = 0, usekernel = FALSE, adjust = 1)




#Naive Bayes--Chi_Sq_2
set.seed(123)
trainControl <- trainControl(method = "cv", number = 10)

NB_model <- train(Class ~ ., data = Chi_Sq_2, method = "naive_bayes",
                  trControl = trainControl)
NB_model



#Best Laplace=0,usekernel = TRUE and adjust = 1.
US_Best_NB_Chi_Sq_2 <- naiveBayes(x, y, laplace = 0, usekernel = TRUE, adjust = 1)



#Naive Bayes--RF_2
set.seed(123)
trainControl <- trainControl(method = "cv", number = 10)

NB_model <- train(Class ~ ., data = RF_2, method = "naive_bayes",
                  trControl = trainControl)
NB_model

#Best Laplace=0,usekernel = FALSE and adjust = 1.
US_Best_NB_RF_2 <- naiveBayes(x, y, laplace = 0, usekernel = TRUE, adjust = 1)

#######################################################################################

#SVM

#SVM--Anov_F_2
set.seed(123)
subset_size <- 100

sampled_indices <- sample(nrow(Anov_F_2), size = subset_size)
subset_dataset <- Anov_F_2[sampled_indices, ]

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

SVM_model <- caret::train(Class ~ ., data = subset_dataset, method = "svmRadial",
                          preProc = c("center", "scale"),
                          trControl = train_control, tuneGrid = svmGrid)
SVM_model


features <- subset_dataset[, -which(names(subset_dataset) == "Class")]
target <- subset_dataset$Class
gamma_value <- 1 / (2 * 0.4^2)


#Best Value sigma = 0.4 and C = 1:
US_Best_SVM_Anov_F_2 <- ksvm(x = as.matrix(features), 
                             y = as.factor(target), 
                             type = "C-svc", 
                             kernel = "rbfdot", 
                             C = 1.0, 
                             kpar = list(sigma = 0.4), 
                             prob.model = TRUE)




#SVM--Chi_Sq_2
set.seed(123)
subset_size <- 100

sampled_indices <- sample(nrow(Chi_Sq_2), size = subset_size)
subset_dataset <- Chi_Sq_2[sampled_indices, ]

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

SVM_model <- caret::train(Class ~ ., data = subset_dataset, method = "svmRadial",
                          preProc = c("center", "scale"),
                          trControl = train_control, tuneGrid = svmGrid)
SVM_model


features <- subset_dataset[, -which(names(subset_dataset) == "Class")]
target <- subset_dataset$Class
gamma_value <- 1 / (2 * 0.4^2)


#Best Value sigma = 0.15 and C = 1.6:
US_Best_SVM_Chi_Sq_2 <- ksvm(x = as.matrix(features), 
                             y = as.factor(target), 
                             type = "C-svc", 
                             kernel = "rbfdot", 
                             C = 1.6, 
                             kpar = list(sigma = 0.15), 
                             prob.model = TRUE)





#SVM--RF_2
set.seed(123)
subset_size <- 100

sampled_indices <- sample(nrow(RF_2), size = subset_size)
subset_dataset <- RF_2[sampled_indices, ]

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.05), C = seq(1.0, 2.0, by = 0.1))

SVM_model <- caret::train(Class ~ ., data = subset_dataset, method = "svmRadial",
                          preProc = c("center", "scale"),
                          trControl = train_control, tuneGrid = svmGrid)
SVM_model


features <- subset_dataset[, -which(names(subset_dataset) == "Class")]
target <- subset_dataset$Class
gamma_value <- 1 / (2 * 0.4^2)


#Best Value sigma = 0.4 and C = 1:
US_Best_SVM_RF_2 <- ksvm(x = as.matrix(features), 
                         y = as.factor(target), 
                         type = "C-svc", 
                         kernel = "rbfdot", 
                         C = 1.0, 
                         kpar = list(sigma = 0.4), 
                         prob.model = TRUE)


#########################################################################################################
##Random Forest
#Random Forest--Anov_F_2

set.seed(123)
subset_size <- 300
sampled_indices <- sample(nrow(Anov_F_2), size = subset_size)
subset <- Anov_F_2[sampled_indices, ]
subset$Class <- factor(subset$Class)
clean_levels <- make.names(levels(subset$Class), unique = TRUE)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(subset)-1, by = 1)




# Reassign cleaned levels back to the factor
levels(subset$Class) <- clean_levels

# Model Training
rfFit <- train(x = subset[,-13], y = subset$Class, method = "rf",
               ntree = 500, tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE, metric = "ROC", trControl = ctrl)
rfFit


# Instantiate the Random Forest best model with mtry = 5
US_Best_RF_Anov_F_2 <- randomForest(Class ~ ., data = subset, mtry = 5, ntree = 500)





#Random Forest--Chi_Sq_2
set.seed(123)
subset_size <- 300
sampled_indices <- sample(nrow(Chi_Sq_2), size = subset_size)
subset <- Chi_Sq_2[sampled_indices, ]
subset$Class <- factor(subset$Class)
clean_levels <- make.names(levels(subset$Class), unique = TRUE)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(subset)-1, by = 1)




# Reassign cleaned levels back to the factor
levels(subset$Class) <- clean_levels

# Model Training
rfFit <- train(x = subset[,-13], y = subset$Class, method = "rf",
               ntree = 500, tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE, metric = "ROC", trControl = ctrl)
rfFit




# Instantiate the Random Forest best model with mtry = 3
US_Best_RF_Chi_Sq_2 <- randomForest(Class ~ ., data = subset, mtry = 3, ntree = 500)








#Random Forest--RF_2

set.seed(123)
subset_size <- 300
sampled_indices <- sample(nrow(RF_2), size = subset_size)
subset <- RF_2[sampled_indices, ]
subset$Class <- factor(subset$Class)
clean_levels <- make.names(levels(subset$Class), unique = TRUE)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(subset)-1, by = 1)




# Reassign cleaned levels back to the factor
levels(subset$Class) <- clean_levels

# Model Training
rfFit <- train(x = subset[,-13], y = subset$Class, method = "rf",
               ntree = 500, tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE, metric = "ROC", trControl = ctrl)
rfFit



# Instantiate the Random Forest best model with mtry = 7
US_Best_RF_RF_2 <- randomForest(Class ~ ., data = subset, mtry = 7, ntree = 500)




##################################################################################################

## J48 Decision Tree ##
perform_J48 <- function(test_data, data_name){
  
  # test the model on the test dataset
  pred <- predict(OS_Best_DecisionTree_Anov_F_1, newdata = test_data, type = "class")
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$model = c("J48")
  results$data_name = c(data_name)
  return (results)
}



perform_J48(Anov_F_1_test, Anov_F_1_test)













## Naive Bayes ##
perform_NB <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class)
  test_data$Class <- factor(test_data$Class)
  
  # build a Naïve Bayes model from training dataset
  nb_classify <- naiveBayes(Class ~ ., data=training_data)
  # test on test dataset
  pred <- predict(nb_classify, newdata = test_data, type = "class")
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$model = c("NB")
  results$data_name = c(data_name)
  return (results)
}

















## C5.0 Decision Tree ##
perform_C5 <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class)
  test_data$Class <- factor(test_data$Class)
  
  C5_tree <- C5.0(Class ~ ., data = training_data)
  
  # test 
  pred <- predict(C5_tree, newdata = test_data, type = "class")
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$model = c("C5")
  results$data_name = c(data_name)
  return (results)
}












## Logistic Regression ##
perform_LogR <- function(training_data, test_data, data_name){
  
  training_data$Class <- factor(training_data$Class)
  test_data$Class <- factor(test_data$Class)
  
  # set up binomial logisitic regression model
  logitModel <- glm(Class ~ ., data = training_data, family = "binomial") 
  
  # use predict() with type = "response" to compute predicted probabilities. 
  logitModel.pred <- predict(logitModel, test_data, type = "response")
  
  # performance measures on the test dataset
  pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
  
  # compute confusion matrix
  cm  <- confusionMatrix(data=pred, reference = test_data$Class)
  
  #save results as a data frame
  results = data.frame(cbind(t(cm$overall),t(cm$byClass)))
  results$model = c("LogReg")
  results$data_name = c(data_name)
  return (results)
}






















################################################################################
### Evaluation ###


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
  J48_results = perform_J48(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])
  
  NB_results = perform_NB(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])
  
  C5_results = perform_C5(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])
  
  LogR_results = perform_LogR(oversample_set[[i]], oversample_test_set[[i]], oversample_set_names[i])
  
  over_sample_output_df = rbind(over_sample_output_df, J48_results, NB_results, C5_results, LogR_results)
  
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
  
  LogR_results = perform_LogR(undersample_set[[i]], undersample_test_set[[i]], undersample_set_names[i])
  
  under_sample_output_df = rbind(under_sample_output_df, J48_results, NB_results, C5_results, LogR_results)
  
}

# save results
under_sample_output_df <- apply(under_sample_output_df,2,as.character)
write.csv(under_sample_output_df, "under_sample_results.csv")






calculate_performance_measures <- function(pred, actual) {
  cm <- confusionMatrix(data = pred, reference = actual)
  
  # Calculate TP rate (Sensitivity) and TN rate (Specificity)
  TPR <- cm$byClass["Sensitivity"]
  TNR <- cm$byClass["Specificity"]
  
  # Calculate other metrics
  FP_rate <- 1 - TNR
  precision <- cm$byClass["Pos Pred Value"]
  recall <- TPR # Recall is the same as Sensitivity/TPR
  F_measure <- cm$byClass["F1"]
  ROC_area <- cm$byClass["ROC"]
  MCC <- cm$overall["MCC"]
  kappa <- cm$overall["Kappa"]
  
  # Create a dataframe to store performance measures
  performance_table <- data.frame(
    TPR = TPR,
    TNR = TNR,
    FP_rate = FP_rate,
    precision = precision,
    recall = recall,
    F_measure = F_measure,
    ROC_area = ROC_area,
    MCC = MCC,
    kappa = kappa
  )
  
  return(performance_table)
}




# Test your own model on new data (replace this with your actual model)
predicted_classes <- predict(US_Best_Knn_Anov_F_2, newdata = Anov_F_2_test)

# Convert factor predictions to numeric and then adjust to 0-based indexing
predicted_classes <- as.integer(predicted_classes) - 1


# Assuming 'actual_classes' contains the true class labels for the test data
actual_classes <- Anov_F_1_test$Class


predicted_classes <- factor(predicted_classes)
actual_classes <- factor(actual_classes)



# Calculate performance measures
performance_table <- calculate_performance_measures(predicted_classes, actual_classes)

# Show the performance measures table
print(performance_table)


