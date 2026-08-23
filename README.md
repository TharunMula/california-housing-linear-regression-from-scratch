# California Housing Price Prediction

This is my first Machine Learning project where I implemented **Linear Regression from scratch** to predict house prices using the California Housing dataset.

I built this project while learning the fundamentals of supervised learning and wanted to understand what happens inside a Linear Regression model instead of directly using a pre-built model.

## What I did

I started by splitting the dataset into training and testing data and then scaled the features using `StandardScaler`.

After that, I implemented the Linear Regression training process using:

* Linear Regression
* Gradient Descent
* Cost Function
* Weight and Bias Updates
* L2 Regularization

The basic prediction used in the model is:

`ŷ = Xw + b`

I used Gradient Descent to update the weights and bias repeatedly until the model learned from the training data.

## Model Evaluation

After training the model, I tested it on data that the model had not seen before.

The final test results were:

* **Test RMSE:** 0.753
* **Test R²:** 0.567

The model explains approximately **56.7% of the variation in house prices** in the test data.

I also compared the training and test results:

* **Training R²:** 0.590
* **Test R²:** 0.567
* **Training RMSE:** 0.740
* **Test RMSE:** 0.753

The relatively small difference between the training and test results shows that the model is generalizing reasonably well to unseen data.

## Regularization

I also implemented **L2 regularization** and experimented with different values of λ.

For example:

λ = 1
   ───────────────→  R² = 0.5672 | RMSE = 0.7531

λ = 10
   ───────────────→  R² = 0.5671 | RMSE = 0.7532

λ = 100
   ───────────────→  R² = 0.5662 | RMSE = 0.7539

Increasing the regularization strength did not improve the results for this dataset. This was useful because it showed me that regularization is not automatically going to improve a model; its effect depends on the dataset and the model.

## Actual vs Predicted

I also plotted the actual house prices against the prices predicted by the model to visually understand the prediction errors(avaible in the screenshot)

<!-- Add your graph here -->

## What I learned from this project

This project helped me understand the complete process behind a basic regression model.

Before doing this project, I understood concepts such as Gradient Descent, Cost Function and Regularization mainly from the course. Implementing them myself helped me understand how they work together during model training.

I also learned the importance of evaluating a model on unseen data rather than looking only at its training performance.

This is my **baseline Linear Regression model**, and I plan to use this as a starting point for experimenting with more advanced regression techniques and models.
