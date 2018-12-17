rm(list = ls())

library(tidyverse)
library(randomForest)
library(corrplot)
library(cvTools)
library(leaps)
library(dplyr)
library('MASS')
library(glmnet)
library(ggfortify)
library(boot)

# ------- Working directory -------
all_data = read.csv("~/Desktop/Stanford/Courses/MS&E226/Project/OnlineNewsPopularity/OnlineNewsPopularity.csv", header = TRUE, sep = ",")

# For reproducibility
set.seed(0)

# ---------------------- PREPROCESSING ----------------------

# Remove useless columns
all_data$url <- NULL

# ---------------------- Days of week ----------------------
# Gather variables related to the day of the week in one categorical variable
# Create the day_of_week column:
# 1 -> Monday
# 2 -> Tuesday
# 3 -> Wednesday
# 4 -> Thursday
# 5 -> Friday
# 6 -> Saturday
# 7 -> Sunday
all_data = mutate(all_data, day_of_week =
                    weekday_is_monday +
                    2*weekday_is_tuesday +
                    3*weekday_is_wednesday +
                    4*weekday_is_thursday +
                    5*weekday_is_friday +
                    6*weekday_is_saturday +
                    7*weekday_is_sunday)

all_data$weekday_is_monday <- NULL
all_data$weekday_is_tuesday <- NULL
all_data$weekday_is_wednesday <- NULL
all_data$weekday_is_thursday <- NULL
all_data$weekday_is_friday <- NULL
all_data$weekday_is_saturday <- NULL
all_data$weekday_is_sunday <- NULL
all_data$is_weekend <- NULL
all_data$day_of_week <- factor(all_data$day_of_week)

# ---------------------- Types of Channel ----------------------
# Gather variables related to the type of channel in one categorical variable
# Create the data_channel column:
# 0 -> Undertermined
# 1 -> lifestyle
# 2 -> entertainment
# 3 -> bus
# 4 -> socmed
# 5 -> tech
# 6 -> world
all_data = mutate(all_data, type_of_channel =
                    data_channel_is_lifestyle +
                    2*data_channel_is_entertainment +
                    3*data_channel_is_bus +
                    4*data_channel_is_socmed +
                    5*data_channel_is_tech +
                    6*data_channel_is_world)

all_data$data_channel_is_lifestyle <- NULL
all_data$data_channel_is_entertainment <- NULL
all_data$data_channel_is_bus <- NULL
all_data$data_channel_is_socmed <- NULL
all_data$data_channel_is_tech <- NULL
all_data$data_channel_is_world <- NULL
all_data$type_of_channel <- factor(all_data$type_of_channel)

# ---------------------- Swap of columns ----------------------
all_data[ , c(46,48)] <- all_data[ , c(48,46)]
colnames(all_data)[c(46,48)] <- colnames(all_data)[c(48,46)]

# Check for NA
na_count <-sapply(all_data, function(y) sum(is.na(y)))
na_count <- data.frame(na_count)

# Remove NaNs
l = which(
  all_data$n_tokens_content == 0 |
    all_data$n_tokens_title == 0 |
    all_data$num_keywords == 0 | 
    all_data$average_token_length == 0 |
    all_data$min_negative_polarity == 0 |
    all_data$max_positive_polarity == 0 |
    all_data$avg_negative_polarity == 0 |
    all_data$avg_positive_polarity == 0)
all_data = all_data[-l,]

# Track colinear columns
M <- cor(all_data[sapply(all_data, function(x) !is.factor(x))],use = "everything")
i = 0
for (col.name in colnames(M)) {
  i = i + 1
  j = 0
  for (tmp in M[col.name, ]) {
    j = j + 1
    if ((isTRUE(all.equal(tmp, -1)) | isTRUE(all.equal(tmp, 1))) && (i != j)) {
      cat(colnames(M)[i], colnames(M)[j], "\n")
    }
  }
}

# Remove parallel vectors
all_data$rate_negative_words <- NULL

# Remove vector that are linear combination of other vectors
cols = c("LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04")
sum(all_data[cols]) # == nrows
all_data$LDA_00 <- NULL

# Remove vectors parallel to the intercept
sum(all_data$n_non_stop_words) # == nrows
all_data$n_non_stop_words <- NULL

# Train / Test
set.seed(123)
index_train = sample(nrow(all_data), size = nrow(all_data)*0.80)
train_data = all_data[index_train,]
test_data = all_data[-index_train,]


# ---------------------- MODELLING ----------------------

# ---------------------- Baseline ----------------------
baseline_linear <- lm(shares~., data = train_data)
baeseline_train_MSE = sqrt(mean(baseline_linear$residuals^2))
OLS = cvFit(baseline_linear, data = train_data, y=train_data$shares, K=10)
OLS

plot(fitted(baseline_linear),
     residuals(baseline_linear),
     xlab = "fitted values",
     ylab = "residuals",
     main = "residuals against fitted values")
abline(0,0)


# ---------------------- Log-transform ----------------------
train_data['log_shares'] <- log(train_data['shares']+1)
train_data$shares <- NULL
test_data['log_shares'] <- log(test_data['shares']+1)
test_data$shares <- NULL
baseline_log_linear <- lm(log_shares~., data = train_data)
baeseline_LOG_train_MSE = sqrt(mean(baseline_log_linear$residuals^2))
OLS_log = cvFit(baseline_log_linear, data = train_data, y=train_data$log_shares, K=10)
OLS_log

# Plot residuals for log-transform
plot(fitted(baseline_log_linear),
     residuals(baseline_log_linear),
     xlab = "fitted values",
     ylab = "residuals",
     main = "residuals against fitted values")
abline(0,0)

# ---------------------- Log-transform of the features ----------------------

# Matrix representation + covariance
M <- cor(train_data[sapply(train_data, function(x) !is.factor(x))],use = "everything")
corrplot(M, method = 'circle')

threshold = 0.85
i = 0
for (col.name in colnames(M)) {
  i = i + 1
  j = 0
  for (tmp in M[col.name, ]) {
    j = j + 1
    if ((abs(tmp) > threshold) && (i < j)) {
      cat(colnames(M)[i], colnames(M)[j], "\n")
    }
  }
}
# Correlated features
cols = c(
  "n_unique_tokens",
  "n_non_stop_unique_tokens",
  "kw_min_min",
  "kw_max_max",
  "kw_max_min",
  "kw_avg_min",
  "self_reference_max_shares",
  "self_reference_avg_sharess") 

# See in each group of correlated feature which one has the greater correlation
# with the outcome variable
apply(train_data[, cols], 2, cor, y=train_data$log_shares)

# Remove other correlated features
train_data$n_unique_tokens <- NULL
train_data$kw_max_max <- NULL
train_data$kw_max_min <- NULL
train_data$self_reference_max_shares <- NULL
test_data$n_unique_tokens <- NULL
test_data$kw_max_max <- NULL
test_data$kw_max_min <- NULL
test_data$self_reference_max_shares <- NULL

# Look at the numerical variables
nums <- unlist(lapply(train_data, is.numeric))
# Look at the columns with only positive values
cols = colnames(train_data[, nums][apply(train_data[, nums] > 0, 2, all)])
cols = cols[2:(length(cols) - 1)]
aux.log = log(train_data[,cols])
aux = train_data[,cols]
# Compare the correlation of the feature and its log transform with the outcome
# variable
cols = cols[abs(apply(aux.log, 2, cor, y=train_data$log_shares)) >
              abs(apply(aux, 2, cor, y=train_data$log_shares))]

# Log-transform the features whos correlation with the outcome variable
# is greater after transform
train_data$n_tokens_title.log <- log(train_data$n_tokens_title)
train_data$n_non_stop_unique_tokens.log <- log(train_data$n_non_stop_unique_tokens)
train_data$average_token_length.log <- log(train_data$average_token_length)
train_data$num_keywords.log <- log(train_data$num_keywords)
train_data$global_rate_positive_words.log <- log(train_data$global_rate_positive_words)
train_data$global_rate_negative_words.log <- log(train_data$global_rate_negative_words)
train_data$min_positive_polarity.log <- log(train_data$min_positive_polarity)

train_data$n_tokens_title <- NULL
train_data$n_non_stop_unique_tokens <- NULL
train_data$average_token_length <- NULL
train_data$num_keywords <- NULL
train_data$global_rate_positive_words <- NULL
train_data$global_rate_negative_words <- NULL
train_data$min_positive_polarity <- NULL

test_data$n_tokens_title.log <- log(test_data$n_tokens_title)
test_data$n_non_stop_unique_tokens.log <- log(test_data$n_non_stop_unique_tokens)
test_data$average_token_length.log <- log(test_data$average_token_length)
test_data$num_keywords.log <- log(test_data$num_keywords)
test_data$global_rate_positive_words.log <- log(test_data$global_rate_positive_words)
test_data$global_rate_negative_words.log <- log(test_data$global_rate_negative_words)
test_data$min_positive_polarity.log <- log(test_data$min_positive_polarity)

test_data$n_tokens_title <- NULL
test_data$n_non_stop_unique_tokens <- NULL
test_data$average_token_length <- NULL
test_data$num_keywords <- NULL
test_data$global_rate_positive_words <- NULL
test_data$global_rate_negative_words <- NULL
test_data$min_positive_polarity <- NULL

# Fit a linear model with these new features
linear.log <- lm(log_shares ~ ., data = train_data)
linear.log.rmse = sqrt(mean(linear.log$residuals^2))
# Cross validation for test error estimation
linear.log.cv = cvFit(linear.log,
                      data=train_data,
                      y=train_data$log_shares,
                      K=10,
                      seed=123)
linear.log.rmse
linear.log.cv

# ---------------------- Feature selection ----------------------

# AIC forward-backward selection
modelAIC <- lm(log_shares~., data = train_data)
step <- stepAIC(modelAIC, direction = 'both')
summary(step)

linear.log.AIC.rmse = sqrt(mean(step$residuals^2))
linear.log.AIC.cv = cvFit(step,
                          data=train_data,
                          y=train_data$log_shares,
                          K=10,
                          seed=123)
linear.log.AIC.rmse
linear.log.AIC.cv

# Columns selected by AIC
cols.AIC = c(
  "log_shares",
  "timedelta",
  "num_hrefs",
  "num_self_hrefs",
  "num_imgs",
  "kw_min_min",
  "kw_avg_min",
  "kw_min_max",
  "kw_avg_max",
  "kw_min_avg",
  "kw_max_avg",
  "kw_avg_avg",
  "self_reference_min_shares",
  "self_reference_avg_sharess",
  "LDA_01",
  "LDA_02",
  "LDA_03",
  "LDA_04",
  "global_subjectivity",
  "global_sentiment_polarity",
  "rate_positive_words",
  "title_subjectivity",
  "title_sentiment_polarity",
  "abs_title_subjectivity",
  "type_of_channel",
  "day_of_week",
  "n_tokens_title.log",
  "n_non_stop_unique_tokens.log",
  "average_token_length.log",
  "num_keywords.log",
  "global_rate_positive_words.log",
  "global_rate_negative_words.log",
  "min_positive_polarity.log")

# Lasso selection
lambdas = 10^seq(-10,-2, 0.1)
train_Lasso <- train_data[,cols.AIC]
train_Lasso$log_shares <- NULL
fm.lasso <- glmnet(as.matrix(train_Lasso), as.double(train_data$log_shares), alpha=1, standardize = TRUE, thresh = 1e-12)
plot(fm.lasso, xvar ="lambda")

# Random Forest
# To unquote, takes ~10min
#index_RF = sample(nrow(train_data[,cols.AIC]), size = nrow(train_data[,cols.AIC])*0.60)
#train_data[index_RF,cols.AIC]
#fit=randomForest(log_shares~., data=train_data[index_RF,cols.AIC])
#varImpPlot(fit)

# ------ Interaction with the categorical variables ------
linear.log.AIC.interaction = lm(log_shares ~ . +
                                  day_of_week:. +
                                  type_of_channel:. -
                                  day_of_week:type_of_channel,
                                data=train_data[,cols.AIC])

# Cross validation for test error estimation
linear_final_model_Train_CV = cvFit(linear.log.AIC.interaction,
                                      data=train_data[, cols.AIC],
                                      y=train_data$log_shares,
                                      K=10,
                                      seed=123)

################################## Results ##################################

# RMSE on the Train set of the best model
linear_final_model_Train_RMSE = sqrt(mean(linear.log.AIC.interaction$residuals^2))
linear_final_model_Train_RMSE

# RMSE on the CV set of the best model
linear_final_model_Train_CV

# RMSE on the test set of the best model
linear_final_model_Test_RMSE = sqrt(mean((predict(linear.log.AIC.interaction, test_data[,cols.AIC]) - test_data[, cols.AIC]$log_shares)**2))
linear_final_model_Test_RMSE



# ------------ Inference ------------
# Train Data
final_model_inf_train <- lm(log_shares ~ . +
     day_of_week:. +
     type_of_channel:. -
     day_of_week:type_of_channel,
   data=train_data[,cols.AIC])
summary(final_model_inf_train)

idx <- order(coef(summary(final_model_inf_train))[,4])  # sort out the p-values
out <- coef(summary(final_model_inf_train))[idx,]       # reorder coef, SE, etc. by increasing p
out

count_at99 <-(out[,4]<=0.01)
table(count_at99)["FALSE"]
table(count_at99)["TRUE"]/(table(count_at99)["FALSE"]+table(count_at99)["TRUE"])

count_at95 <-(out[,4]<=0.05)
table(count_at95)["FALSE"]
table(count_at95)["TRUE"]/(table(count_at95)["FALSE"]+table(count_at95)["TRUE"])

exp(mean(train_data$log_shares))

count_atBonferroni <-(out[,4]<=0.05/403)
table(count_atBonferroni)["TRUE"]

# Test Data
final_model_inf_test = lm(log_shares ~ . +
                                  day_of_week:. +
                                  type_of_channel:. -
                                  day_of_week:type_of_channel,
                                data=test_data[,cols.AIC])
summary(final_model_inf_test)

idx_test <- order(coef(summary(final_model_inf_test))[,4])  # sort out the p-values
out_test <- coef(summary(final_model_inf_test))[idx_test,]       # reorder coef, SE, etc. by increasing p
out_test

count_at99_test <-(out_test[,4]<=0.01)
table(count_at99_test)["FALSE"]
table(count_at99_test)["TRUE"]/(table(count_at99_test)["FALSE"]+table(count_at99_test)["TRUE"])

count_at95_test <-(out_test[,4]<=0.05)
table(count_at95_test)["FALSE"]
table(count_at95_test)["TRUE"]/(table(count_at95_test)["FALSE"]+table(count_at95_test)["TRUE"])

exp(mean(train_data$log_shares))

# All covariates on training data
final_model_inf_train_all <- lm(log_shares ~ . +
                              day_of_week:. +
                              type_of_channel:. -
                              day_of_week:type_of_channel,
                            data=train_data)
summary(final_model_inf_train_all)

idx_all <- order(coef(summary(final_model_inf_train_all))[,4])  # sort out the p-values
out_all <- coef(summary(final_model_inf_train_all))[idx_all,]       # reorder coef, SE, etc. by increasing p
out_all

count_at99_final <-(out_all[,4]<=0.01)
table(count_at99_final)["FALSE"]
table(count_at99_final)["TRUE"]/(table(count_at99_final)["FALSE"]+table(count_at99_final)["TRUE"])

count_at95_final <-(out_all[,4]<=0.05)
table(count_at95_final)["FALSE"]
table(count_at95_final)["TRUE"]/(table(count_at95_final)["FALSE"]+table(count_at95_final)["TRUE"])

exp(mean(train_data$log_shares))

count_atBonferroni <-(out[,4]<=0.05/403)
table(count_atBonferroni)["TRUE"]

# Median Quotient
median(coef(summary(final_model_inf_train))[,3])/median(coef(summary(final_model_inf_test))[,3])


# ------------ Bootstrap ------------
X <- train_data
X$log_shares <- NULL
Y <- train_data$log_shares
df = data.frame(X,Y)

coef.boot = function(data, indices) {
  fm = lm(log_shares ~ . +
            day_of_week:. +
            type_of_channel:. -
            day_of_week:type_of_channel, data = train_data[indices,cols.AIC])
  return(coef(fm))
}

# Takes avg. 15min
#boot.out = boot(df, coef.boot, 1000)
summary(boot.out)

# Compared to R
fm = lm(log_shares ~ . +
            day_of_week:. +
            type_of_channel:. -
            day_of_week:type_of_channel, data = train_data[,cols.AIC])
summary(fm)

confint(fm, '(Intercept)', level = 0.95)
confint(fm, 'timedelta', level = 0.95)
confint(fm, 'num_hrefs', level = 0.95)
confint(fm, 'num_self_hrefs', level = 0.95)
confint(fm, 'num_imgs', level = 0.95)
confint(fm, 'kw_min_min', level = 0.95)
confint(fm, 'kw_avg_min', level = 0.95)
confint(fm, 'kw_min_max', level = 0.95)

#Plot boxplot
SE_R = coef(summary(fm))[,2]
SE_ <- vector()

for (i in 1:403){
  SE_[i] = SE_R[i]/sd(boot.out$t[,i])
}
boxplot(SE_)
