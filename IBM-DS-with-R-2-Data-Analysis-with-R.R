install.packages("rlang")
install.packages("tidymodels")

# Library for modeling
library(tidymodels)

# Load tidyverse
library(tidyverse)

# Download NOAA Weather Dataset ####
URL = 'https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-sample-data.tar.gz'
download.file(URL, destfile = "noaa-weather-sample-data.tar.gz")

# Untar the zipped file ####
untar("noaa-weather-sample-data.tar.gz")

# Read the raw dataset ####
df <- read.csv("noaa-weather-sample-data/jfk_weather_sample.csv")

# display the first few rows of the dataframe
head(df)

# Take a glimpse of the dataset to see the different column data types and 
#make sure it is the correct subset dataset with about 5700 rows and 9 columns.
glimpse(df)

# Select a subset of data columns and inspect the column types ####
df <- df %>% select(3,4,6,7,9)

# Show the first 10 rows of this new dataframe
head(df,10)

# Inspect the unique values present in the column HOURLYPrecip
unique(df$HOURLYPrecip)

# for the column HOURLYPrecip, replace all the T values with "0.0" and
#remove "s" from values like "0.02s
df <- df %>% mutate(HOURLYPrecip = str_remove(HOURLYPrecip, 
                                              pattern = "s$"),
                    HOURLYPrecip = str_replace(HOURLYPrecip, 
                                               "T", "0.0")) %>%
              replace(is.na(.), 0)

unique(df$HOURLYPrecip)

# check the types of the columns
glimpse(df)

# Convert Columns to Numerical Types ###
df$HOURLYPrecip = as.numeric(df$HOURLYPrecip)

# check the types of the columns
glimpse(df)

# rename columns ###
df <- df %>% rename(relative_humidity=HOURLYRelativeHumidity, 
                    dry_bulb_temp_f=HOURLYDRYBULBTEMPF,
                    precip=HOURLYPrecip,
                    wind_speed=HOURLYWindSpeed,
                    station_pressure=HOURLYStationPressure)

# split the data into training and testing set using 80% of the data for training ####
set.seed(1234)
df_split <- initial_split(df, prop = 0.8)
train_data <- training(df_split)
test_data <- testing(df_split)

# plot histograms of the training variables ####
train_data %>% ggplot(aes(relative_humidity))+geom_histogram(binwidth = 1)
train_data %>% ggplot(aes(dry_bulb_temp_f))+geom_histogram(binwidth = 1)
train_data %>% ggplot(aes(precip))+geom_histogram(binwidth = 1)
train_data %>% ggplot(aes(wind_speed))+geom_histogram(binwidth = 1)
train_data %>% ggplot(aes(station_pressure))+geom_histogram(binwidth = 1)

# create simple linear regression, where precip is the response variable ####
model_humid <- lm(precip ~ relative_humidity, data = train_data)
summary(model_humid)
train_data %>% ggplot(aes(x=relative_humidity, y=precip))+geom_point()

model_dry <- lm(precip ~ dry_bulb_temp_f, data = train_data)
train_data %>% ggplot(aes(x=dry_bulb_temp_f, y=precip))+geom_point()

model_wind <- lm(precip ~ wind_speed, data = train_data)
summary(model_wind)
train_data %>% ggplot(aes(x=wind_speed, y=precip))+geom_point()

model_pressure <- lm(precip ~ station_pressure, data = train_data)
summary(model_pressure)
train_data %>% ggplot(aes(x=station_pressure, y=precip))+geom_point()

# improve the model ####
# 1. Multiple Linear Regression
mlr <- lm(precip~relative_humidity+wind_speed+station_pressure,
          data = train_data)
summary(mlr)

# 2. Polynomial Regression
poly_reg <- lm(precip~poly(relative_humidity, 3),data = train_data)
summary(poly_reg)
# plot(precip~relative_humidity, data = train_data)
# lines(sort(train_data$relative_humidity),                 # Draw polynomial regression curve
#       fitted(poly_reg)[order(train_data$relative_humidity)],
#       col = "red",
#       type = "l")

ggplot(train_data, aes(x = relative_humidity, y = precip)) + 
  geom_point() + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, 3), 
              col = "red", se = FALSE) 

library(glmnet)
# 3. Regularizaton
df_recipe <- recipe(precip ~ ., data = train_data)
# Ridge (L2) regularization
ridge_spec <- linear_reg(penalty = 0.1, mixture = 0) %>% set_engine("glmnet")

ridge_wf <- workflow() %>% add_recipe(df_recipe)

ridge_fit <- ridge_wf %>% add_model(ridge_spec) %>% fit(data = train_data)

ridge_fit %>% pull_workflow_fit() %>% tidy()

# find the best model, using matrices (MSE, RMSE or R-squared) ####
train_fit_lm <- linear_reg() %>% set_engine("lm") %>% 
  fit(precip~relative_humidity+wind_speed+station_pressure,
      data = train_data)

train_fit_poly <- linear_reg() %>% set_engine("lm") %>% 
  fit(precip ~ poly(relative_humidity, 3),
      data = train_data)

train_fit_ridge <- linear_reg(penalty = 0.1, mixture = 0) %>% set_engine("glmnet") %>% 
  fit(precip~relative_humidity+wind_speed+station_pressure,
      data = train_data)

train_results_lm <- train_fit_lm %>% predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)

test_results_lm <- train_fit_lm %>% predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)

train_results_poly <- train_fit_poly %>% predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)

test_results_poly <- train_fit_poly %>% predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)

train_results_ridge <- train_fit_ridge %>% predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)

test_results_ridge <- train_fit_ridge %>% predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)

## visualize the model using training and testing set
test_results_lm %>%
  mutate(train = "testing") %>%
  bind_rows(train_results_lm %>% mutate(train = "training")) %>%
  ggplot(aes(truth, .pred)) +
  geom_abline(lty = 2, color = "orange", 
              size = 1.5) +
  geom_point(color = '#006EA1', 
             alpha = 0.5) +
  facet_wrap(~train) +
  labs(x = "Truth", 
       y = "Predicted Precipitation")

test_results_poly %>%
  mutate(train = "testing") %>%
  bind_rows(train_results_poly %>% mutate(train = "training")) %>%
  ggplot(aes(truth, .pred)) +
  geom_abline(lty = 2, color = "orange", 
              size = 1.5) +
  geom_point(color = '#006EA1', 
             alpha = 0.5) +
  facet_wrap(~train) +
  labs(x = "Truth", 
       y = "Predicted Precipitation")

test_results_ridge %>%
  mutate(train = "testing") %>%
  bind_rows(train_results_ridge %>% mutate(train = "training")) %>%
  ggplot(aes(truth, .pred)) +
  geom_abline(lty = 2, color = "orange", 
              size = 1.5) +
  geom_point(color = '#006EA1', 
             alpha = 0.5) +
  facet_wrap(~train) +
  labs(x = "Truth", 
       y = "Predicted Precipitation")

train_error_lm <- rmse(train_results_lm, truth = truth,
                       estimate = .pred)

test_error_lm <- rmse(test_results_lm, truth = truth,
                      estimate = .pred)

train_error_poly <- rmse(train_results_poly, truth = truth,
                         estimate = .pred)

test_error_poly <- rmse(test_results_poly, truth = truth,
                        estimate = .pred)

train_error_ridge <- rmse(train_results_ridge, truth = truth,
                          estimate = .pred)

test_error_ridge <- rmse(test_results_ridge, truth = truth,
                         estimate = .pred)

train_rsq_lm <- rsq(train_results_lm, truth = truth,
                    estimate = .pred)

test_rsq_lm <- rsq(test_results_lm, truth = truth,
                   estimate = .pred)

train_rsq_poly <- rsq(train_results_poly, truth = truth,
                      estimate = .pred)

test_rsq_poly <- rsq(test_results_poly, truth = truth,
                     estimate = .pred)

train_rsq_ridge <- rsq(train_results_ridge, truth = truth,
                       estimate = .pred)

test_rsq_ridge <- rsq(test_results_ridge, truth = truth,
                      estimate = .pred)

model_names <- c("mlr", "poly", "ridge_L2")
train_error <- c(train_error_lm$.estimate, train_error_poly$.estimate, train_error_ridge$.estimate)
test_error <- c(test_error_lm$.estimate, test_error_poly$.estimate, test_error_ridge$.estimate)
train_rsq <- c(train_rsq_lm$.estimate, train_rsq_poly$.estimate, train_rsq_ridge$.estimate)
test_rsq <- c(test_rsq_lm$.estimate, test_rsq_poly$.estimate, test_rsq_ridge$.estimate)
comparison_df <- data.frame(model_names, train_error, test_error, train_rsq, test_rsq)
comparison_df