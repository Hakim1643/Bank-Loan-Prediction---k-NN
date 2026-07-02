############################################################
# Assignment 2: Predicting Bank Customers using kNN
############################################################


##############################
# 1. LOAD REQUIRED LIBRARIES
##############################

library(class)    # For knn()
library(kknn)     # For weighted kNN
library(caret)    # For cross-validation


##############################
# 2. LOAD DATASET
##############################

# Skip first 3 blank rows and load dataset
data <- read.csv(file.choose(), skip = 3, stringsAsFactors = FALSE)

# View dataset structure
head(data)
str(data)


##############################
# 3. DATA PREPROCESSING
##############################

# Remove irrelevant columns
data$ID <- NULL
data$ZIP.Code <- NULL

# Convert target variable to factor
data$Personal.Loan <- as.factor(data$Personal.Loan)


##############################
# 4. DATA VISUALIZATION
##############################

# Histogram: Income distribution
hist(data$Income,
     col = "lightblue",
     main = "Income Distribution",
     xlab = "Income")

# Boxplot: Income vs Personal Loan
boxplot(Income ~ Personal.Loan,
        data = data,
        col = c("lightblue", "orange"),
        main = "Income vs Personal Loan")

# Bar chart: Loan distribution
loan_count <- table(data$Personal.Loan)
barplot(loan_count,
        names.arg = c("No Loan", "Loan"),
        col = c("lightblue", "orange"),
        main = "Loan Acceptance Distribution")


##############################
# 5. FEATURE SELECTION
##############################

# Select important predictors
data2 <- data[, c("Age", "Income", "Family", "CCAvg", "Education", "Mortgage", "Personal.Loan")]


##############################
# 6. DATA NORMALIZATION
##############################

# Function to normalize data
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Apply normalization (excluding target variable)
data_norm <- as.data.frame(lapply(data2[,-7], normalize))

# Add target variable back
data_norm$Personal.Loan <- data2$Personal.Loan


##############################
# 7. DATA SPLITTING (60/40)
##############################

set.seed(123)

# 60% training, 40% validation
sample_rows <- sample(1:nrow(data_norm), 0.6 * nrow(data_norm))

train <- data_norm[sample_rows, ]
valid <- data_norm[-sample_rows, ]


##############################
# 8. FIND BEST K VALUE
##############################

accuracy <- c()

# Test k from 1 to 20
for(i in 1:20) {
  
  pred <- knn(train[,-7],
              valid[,-7],
              train$Personal.Loan,
              k = i)
  
  accuracy[i] <- mean(pred == valid$Personal.Loan)
}

# Determine best k
best_k <- which.max(accuracy)
best_k

# Plot K vs Accuracy
plot(1:20, accuracy,
     type = "o",
     col = "blue",
     xlab = "K value",
     ylab = "Accuracy",
     main = "Choosing Best K")


##############################
# 9. FINAL MODEL (EUCLIDEAN)
##############################

# Apply kNN using best k
best_pred <- knn(train[,-7],
                 valid[,-7],
                 train$Personal.Loan,
                 k = best_k)

# Accuracy
mean(best_pred == valid$Personal.Loan)

# Confusion matrix
table(Predicted = best_pred, Actual = valid$Personal.Loan)


##############################
# 10. WEIGHTED KNN (kknn)
##############################

# Apply weighted kNN
model <- kknn(Personal.Loan ~ .,
              train,
              valid,
              k = best_k,
              kernel = "triangular")

# Predictions
pred2 <- fitted(model)

# Accuracy
mean(pred2 == valid$Personal.Loan)

# Confusion matrix
table(Predicted = pred2, Actual = valid$Personal.Loan)


##############################
# 11. CROSS-VALIDATION (10-FOLD)
##############################

# Define cross-validation method
control <- trainControl(method = "cv", number = 10)

# Train model
model_cv <- train(Personal.Loan ~ .,
                  data = train,
                  method = "knn",
                  trControl = control,
                  tuneLength = 10)

# View results
model_cv

# Best cross-validation accuracy
max(model_cv$results$Accuracy)


##############################
# END OF SCRIPT
##############################