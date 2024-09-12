library(readr)
library(dplyr)
library(ggplot2)

# Load the dataset
data <- read_csv("E:\\BU\\2024summer\\CS555\\Term-Project\\survey lung cancer.csv")

# Clean column names (remove trailing spaces)
colnames(data) <- trimws(colnames(data))

# Convert categorical variables to factors
data$GENDER <- factor(data$GENDER, levels = c("M", "F"))
data$LUNG_CANCER <- factor(data$LUNG_CANCER, levels = c("NO", "YES"))

# Age distribution of the respondents
ggplot(data, aes(x = AGE)) + 
  geom_histogram(binwidth = 5, fill = "lightblue", color = "black") +
  labs(title = "Age Distribution of Respondents", x = "Age", y = "Frequency")

summary(data$AGE)

# Gender distribution among respondents
ggplot(data, aes(x = GENDER, fill = GENDER)) + 
  geom_bar() +
  labs(title = "Gender Distribution", x = "Gender", y = "Count") +
  scale_fill_manual(values = c("lightblue", "pink"))



# Fit the logistic regression model with all predictors
model <- glm(LUNG_CANCER ~ SMOKING + AGE + GENDER + YELLOW_FINGERS + ANXIETY +
               PEER_PRESSURE + `CHRONIC DISEASE` + FATIGUE + ALLERGY + WHEEZING + 
               `ALCOHOL CONSUMING` + COUGHING + `SHORTNESS OF BREATH` +
               `SWALLOWING DIFFICULTY` + `CHEST PAIN`, 
             data = data, family = binomial)

# Display the summary of the model
summary(model)
