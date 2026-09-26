# Spotify User Churn Prediction

## Custom Logistic Regression and Class-Imbalance Analysis

This project investigates binary classification for predicting user churn on a music-streaming platform. A logistic regression classifier is implemented with NumPy and trained using gradient descent. The effect of class imbalance is then examined through random oversampling, and the results are compared with Scikit-learn's class-weighted `LogisticRegression` model.

## Problem Definition

User churn refers to customers discontinuing their use of a service. The objective of this project is to predict the binary target variable `is_churned` from user demographics, subscription information, device attributes, and listening behavior.

The target classes are interpreted as:

- `0` — the user did not churn
- `1` — the user churned

The project evaluates whether a custom logistic regression implementation can identify churned users and explores how balancing the training data changes its predictions.

## Data Preparation

The identifier column `user_id` is excluded from the model inputs, and `is_churned` is used as the classification target.

### Missing Values

Rows containing missing values are removed before model training. The target values are selected using the retained row indices so that the features and labels remain aligned.

### Numerical Features

The following numerical variables are standardized using `StandardScaler`:

- `age`
- `listening_time`
- `songs_played_per_day`
- `skip_rate`
- `ads_listened_per_week`
- `offline_listening`

Standardization places the numerical features on comparable scales before gradient-based optimization.

### Categorical Features

The following categorical variables are transformed using one-hot encoding:

- `gender`
- `country`
- `subscription_type`
- `device_type`

The first category of each feature is dropped to avoid redundant dummy variables.

### Outlier Handling

No explicit outlier-detection or outlier-removal procedure is applied in the saved implementation. Numerical standardization changes feature scale but does not remove extreme observations.

### Train–Test Split

The processed data is divided into training and test subsets. Stratification is applied using the target variable to preserve the original churn-class distribution in both subsets.

## Custom Logistic Regression

The classifier is implemented using NumPy rather than an existing classification estimator. Its main components are the sigmoid activation function, binary cross-entropy objective, gradient calculation, and iterative gradient-descent updates.

### Sigmoid Function

The sigmoid function converts the linear model output into a probability between zero and one:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

where:

$$
z=\mathbf{X}\mathbf{w}+b
$$

Predicted probabilities greater than or equal to `0.5` are assigned to the churn class.

### Class Weights

Class weights are calculated from the number of negative and positive training examples:

$$
w_0=\frac{m}{2n_0}, \qquad w_1=\frac{m}{2n_1}
$$

where $m$ is the number of training samples and $n_0$ and $n_1$ are the numbers of samples in the two classes.

These weights are incorporated into the reported binary cross-entropy cost so that errors on the minority class receive greater importance.

### Weighted Binary Cross-Entropy

The monitored objective is:

$$
J=-\frac{1}{m}\sum_{i=1}^{m}\left[w_1y_i\log(\hat{y}_i)+w_0(1-y_i)\log(1-\hat{y}_i)\right]
$$

A small constant is added inside the logarithms to prevent numerical errors when predicted probabilities approach zero or one.

### Gradient Descent

Weights are initialized to zero and updated over 2,000 epochs with a learning rate of `0.01`:

$$
\mathbf{w}\leftarrow\mathbf{w}-\alpha\frac{\partial J}{\partial\mathbf{w}},
\qquad
b\leftarrow b-\alpha\frac{\partial J}{\partial b}
$$

The cost is printed every 100 epochs to monitor the training process.

In the saved implementation, the weighted residual is calculated inside the gradient function, but the returned gradient uses the unweighted residual `yHat - y`. Consequently, the class weights affect the displayed cost but are not fully reflected in the parameter updates of the custom model. This behavior is consistent with the first experiment's increasing weighted cost and single-class predictions.

## Experiment 1 — Original Imbalanced Data

The custom logistic regression model is first trained on the original stratified training data.

### Results

| Metric | Score |
|---|---:|
| Accuracy | 0.7406 |
| Precision | 0.0000 |

### Confusion Matrix

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 1,465 | 0 |
| Actual 1 | 513 | 0 |

Although the model achieved approximately **74.1% accuracy**, it predicted every test sample as class `0`. It therefore failed to identify any churned users, producing zero precision for the positive class.

The weighted cost also increased from `0.6931` to approximately `0.8209` during training. The high accuracy is therefore misleading and primarily reflects the dominance of non-churned users in the test data.

## Experiment 2 — Random Oversampling

`RandomOverSampler` is applied to the training set to increase the representation of the minority class. The test set remains unchanged so that evaluation is performed on the original test distribution.

The custom logistic regression model is retrained on the resampled data using the same learning rate, number of epochs, and probability threshold.

### Results

| Metric | Score |
|---|---:|
| Accuracy | 0.5025 |
| Precision | 0.2560 |

### Confusion Matrix

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 747 | 718 |
| Actual 1 | 266 | 247 |

After oversampling, the model no longer predicts only the majority class. It correctly identifies 247 churned users, although it also produces 718 false-positive churn predictions.

The weighted cost decreases gradually from `0.6931` to approximately `0.6918`. Overall accuracy falls to approximately **50.3%**, but positive-class precision improves from zero to approximately **25.6%**.

## Experiment 3 — Scikit-learn Logistic Regression

For comparison, Scikit-learn's `LogisticRegression` is trained on the original training data with `class_weight="balanced"` and `max_iter=2000`.

### Results

| Metric | Score |
|---|---:|
| Accuracy | 0.5071 |
| Precision | 0.2722 |

### Confusion Matrix

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 727 | 738 |
| Actual 1 | 237 | 276 |

The Scikit-learn model produces a more balanced distribution of predictions than the initial custom model. Its results are also close to those of the custom model trained on oversampled data.

## Model Comparison

| Model | Accuracy | Precision | True Negatives | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|---:|---:|
| Custom logistic regression — original data | 0.7406 | 0.0000 | 1,465 | 0 | 513 | 0 |
| Custom logistic regression — oversampled data | 0.5025 | 0.2560 | 747 | 718 | 266 | 247 |
| Scikit-learn logistic regression — balanced weights | 0.5071 | 0.2722 | 727 | 738 | 237 | 276 |

The initial model has the highest numerical accuracy but completely ignores the positive class. Balancing the training data substantially changes the behavior of the custom classifier, allowing it to identify churned users at the cost of more false-positive predictions and lower overall accuracy.

The Scikit-learn model achieves the highest positive-class precision and the largest number of correctly identified churned users among the three reported experiments. The comparison demonstrates why accuracy alone is insufficient for evaluating an imbalanced classification problem and why class-level metrics and confusion matrices must also be examined.
