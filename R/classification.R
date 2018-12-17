rm(list = ls())

library(dplyr)
library(pROC)
library(cvTools)
library("MASS")
library(boot)

# Set working directory
setwd("~/Desktop/Part2")

threshold = 1400

# Read the data from file
data.all = read.csv("OnlineNewsPopularity.csv", header = TRUE, sep = ',')

################################## Preprocessing of the data ##################################
# Remove useless columns
data.all$url <- NULL

# Gather variables related to the day of the week in one categorical variable
# Create the day_of_week column:
# 1 -> Monday
# 2 -> Tuesday
# 3 -> Wednesday
# 4 -> Thursday
# 5 -> Friday
# 6 -> Saturday
# 7 -> Sunday
data.all = mutate(data.all, day_of_week =
	weekday_is_monday +
	2*weekday_is_tuesday +
	3*weekday_is_wednesday +
	4*weekday_is_thursday +
	5*weekday_is_friday +
	6*weekday_is_saturday +
	7*weekday_is_sunday)

data.all$weekday_is_monday <- NULL
data.all$weekday_is_tuesday <- NULL
data.all$weekday_is_wednesday <- NULL
data.all$weekday_is_thursday <- NULL
data.all$weekday_is_friday <- NULL
data.all$weekday_is_saturday <- NULL
data.all$weekday_is_sunday <- NULL
data.all$is_weekend <- NULL
data.all$day_of_week <- factor(data.all$day_of_week)

# Gather variables related to the type of channel in one categorical variable
# Create the data_channel column:
# 0 -> Undertermined
# 1 -> lifestyle
# 2 -> entertainment
# 3 -> bus
# 4 -> socmed
# 5 -> tech
# 6 -> world
data.all = mutate(data.all, type_of_channel =
	data_channel_is_lifestyle +
	2*data_channel_is_entertainment +
	3*data_channel_is_bus +
	4*data_channel_is_socmed +
	5*data_channel_is_tech +
	6*data_channel_is_world)

data.all$data_channel_is_lifestyle <- NULL
data.all$data_channel_is_entertainment <- NULL
data.all$data_channel_is_bus <- NULL
data.all$data_channel_is_socmed <- NULL
data.all$data_channel_is_tech <- NULL
data.all$data_channel_is_world <- NULL
data.all$type_of_channel <- factor(data.all$type_of_channel)

# Swap of columns for convenience
data.all[ , c(46,48)] <- data.all[ , c(48,46)]
colnames(data.all)[c(46,48)] <- colnames(data.all)[c(48,46)]

# Check for NA
na_count <-sapply(data.all, function(y) sum(is.na(y)))
na_count <- data.frame(na_count)

# Remove NaNs
l = which(
	data.all$n_tokens_content == 0 |
	data.all$n_tokens_title == 0 |
	data.all$num_keywords == 0 | 
	data.all$average_token_length == 0 |
	data.all$min_negative_polarity == 0 |
	data.all$max_positive_polarity == 0 |
	data.all$avg_negative_polarity == 0 |
	data.all$avg_positive_polarity == 0)

data.all = data.all[-l,]

# Track exactly colinear columns
M <- cor(data.all[sapply(data.all, function(x) !is.factor(x))], use = "everything")
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
data.all$rate_negative_words <- NULL

# Remove vector that are linear combination of other vectors
cols = c("LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04")
sum(data.all[cols]) # == nrows
data.all$LDA_00 <- NULL

# Remove vectors parallel to the intercept
sum(data.all$n_non_stop_words) # == nrows
data.all$n_non_stop_words <- NULL

# Formulate the problem as a classification problem
data.all$shares = as.integer(data.all$shares > threshold)

# Split Train / Test
set.seed(123)
index.train = sample(nrow(data.all), size = nrow(data.all)*0.80)
data.train = data.all[index.train,]
data.test = data.all[-index.train,]

################################## Modeling ##################################

# 0-1 loss cost function
cost_function = function(y, y_hat) {
	mean(as.integer(y_hat > 0.5) != y)
}

#### Baseline logistic regression with all the covariates ####
lr = glm(formula = shares ~ .,
         family = "binomial",
         data = data.train)
lr.error = mean(as.integer(lr$fitted.values > 0.5) != data.train$shares)
lr.error
# Cross validation for test error estimation
lr.cv = cv.glm(
	data=data.train,
	glmfit=lr,
	cost=cost_function,
	K=10)
lr.cv.error = lr.cv$delta[1]
lr.cv.error

#### Feature selection using AIC ####
lr.AIC = step(lr, direction='both')
lr.AIC.error = mean(as.integer(lr.AIC$fitted.values > 0.5) != data.train$shares)
lr.AIC.error
# Cross validation for test error estimation
lr.AIC.cv = cv.glm(
	data=data.train,
	glmfit=lr.AIC,
	cost=cost_function,
	K=10)
lr.AIC.cv.error = lr.AIC.cv$delta[1]
lr.AIC.cv.error
# Columns selected by AIC
cols.AIC = c(
	"shares",
	"timedelta",
	"n_tokens_title",
	"n_tokens_content",
	"n_non_stop_unique_tokens",
	"num_hrefs",
	"num_self_hrefs",
	"average_token_length",
	"num_keywords",
	"kw_min_min",
	"kw_max_min",
	"kw_avg_min",
	"kw_min_max",
	"kw_max_max",
	"kw_avg_max",
	"kw_min_avg",
	"kw_max_avg",
	"kw_avg_avg",
	"self_reference_min_shares",
	"self_reference_max_shares",
	"LDA_01",
	"LDA_02",
	"LDA_03",
	"LDA_04",
	"global_subjectivity",
	"global_rate_positive_words",
	"rate_positive_words",
	"avg_positive_polarity",
	"min_positive_polarity",
	"avg_negative_polarity",
	"title_subjectivity",
	"title_sentiment_polarity",
	"abs_title_subjectivity",
	"type_of_channel",
	"day_of_week")

#### Add interaction with categorical variables ####
lr.AIC.interaction = glm(shares ~ . +
	day_of_week:. +
	type_of_channel:. -
	day_of_week:type_of_channel,
    family = "binomial",
	data=data.train[, cols.AIC])
lr.AIC.interaction.error = mean(as.integer(lr.AIC.interaction$fitted.values > 0.5) != data.train$shares)
lr.AIC.interaction.error
# Cross validation for test error estimation
lr.AIC.interaction.cv = cv.glm(
	data=data.train[, cols.AIC],
	glmfit=lr.AIC.interaction,
	cost=cost_function,
	K=10)
lr.AIC.interaction.cv.error = lr.AIC.interaction.cv$delta[1]
lr.AIC.interaction.cv.error


################################## Results ##################################
# Predictions of the best model and the baseline on the test set
pred.best = as.numeric(predict(lr.AIC.interaction, data.test[, cols.AIC]) >= 0)
pred.baseline = as.numeric(predict(lr, data.test) >= 0)

# Ground truth
truth = data.test$shares

# Compute 0-1 loss for best model and baseline
mean(as.numeric(pred.best != data.test$shares))
mean(as.numeric(pred.baseline != data.test$shares))

# Plot the ROC curve and compute AUC
prob.baseline = predict(lr, data.test, type=c("response"))
prob.best = predict(lr.AIC.interaction, data.test[, cols.AIC], type=c("response"))
data.test$prob.baseline = prob.baseline
data.test$prob.best = prob.best

g.baseline <- roc(shares ~ prob.baseline, data = data.test)
g.best <- roc(shares ~ prob.best, data = data.test)

plot(g.baseline, col="red")
plot(g.best, col="blue", add=TRUE)
auc(g.best)
auc(g.baseline)