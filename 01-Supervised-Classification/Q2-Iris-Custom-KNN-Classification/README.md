# Custom K-Nearest Neighbors Classification

## Iris Classification with a NumPy-Based KNN Implementation

This project implements the **K-Nearest Neighbors (KNN)** classification algorithm using NumPy and evaluates it on the **Iris dataset**. The classifier supports multiple distance metrics, weighted and unweighted voting, and compatibility with Scikit-learn's model-selection tools.

The implementation is combined with feature standardization, stratified cross-validation, and grid search to identify the best-performing KNN configuration.

## Problem Definition

The objective is to classify Iris flowers into one of three species based on four numerical measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The possible target classes are:

- `setosa`
- `versicolor`
- `virginica`

This is a **multiclass classification problem** in which the KNN algorithm predicts the class of each sample according to the labels of its nearest training observations.

## Dataset

The project uses the Iris dataset provided by Scikit-learn. It contains **150 samples**, with **50 observations per class**.

| Property | Value |
|---|---:|
| Number of samples | 150 |
| Number of input features | 4 |
| Number of target classes | 3 |
| Samples per class | 50 |

Because KNN relies directly on distances between samples, feature scaling is applied before classification so that features with larger numerical ranges do not dominate the distance calculations.

## Custom KNN Implementation

The `MyKNNClassifier` class is implemented using NumPy and follows the Scikit-learn estimator interface by inheriting from `BaseEstimator` and `ClassifierMixin`.

The classifier accepts three main parameters:

| Parameter | Description |
|---|---|
| `n_neighbors` | Number of nearest neighbors considered for each prediction |
| `metric` | Distance function used to compare samples |
| `weighted` | Determines whether neighbor votes have equal or distance-based weights |

### Training Phase

The `fit(X, Y)` method stores the training features and labels. It also extracts the unique class labels and converts the training labels into numerical class indices used during voting.

As a non-parametric algorithm, KNN does not learn an explicit mathematical model during training. Instead, prediction is based directly on the stored training samples.

### Distance Calculation

The classifier supports two distance metrics:

#### Euclidean Distance

Euclidean distance measures the straight-line distance between two feature vectors:

$$
d(\mathbf{x},\mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

The implementation calculates the Euclidean distance matrix through vectorized NumPy operations.

#### Manhattan Distance

Manhattan distance is calculated as the sum of the absolute differences between corresponding features:

$$
d(\mathbf{x},\mathbf{y}) = \sum_{i=1}^{n}|x_i-y_i|
$$

### Neighbor Selection

For each test sample, `np.argpartition` identifies the indices of the `k` closest training observations without fully sorting the entire distance matrix.

### Voting Strategies

The classifier supports two voting methods:

- **Unweighted voting:** every selected neighbor contributes one equal vote.
- **Weighted voting:** closer neighbors receive greater influence through inverse-distance weighting.

For weighted voting, the weight assigned to each neighbor is calculated as:

$$
w_i = \frac{1}{d_i + 10^{-12}}
$$

The small constant prevents division by zero when a query sample has zero distance from a training sample. Votes are accumulated for each class, and the class with the highest total vote is returned as the prediction.

## Preprocessing and Evaluation Pipeline

The custom classifier is placed inside a Scikit-learn `Pipeline` with `StandardScaler`:

1. `StandardScaler` standardizes the four input features.
2. `MyKNNClassifier` performs classification using the transformed data.

Keeping preprocessing and classification in the same pipeline ensures that scaling is fitted separately within each cross-validation training fold.

Model performance is evaluated using `StratifiedKFold`. Stratification preserves the class distribution across the five folds, which is appropriate for the three equally represented Iris classes.

## Hyperparameter Optimization

`GridSearchCV` evaluates combinations of the following parameters:

| Hyperparameter | Candidate values |
|---|---|
| Number of neighbors | `1`, `2`, `3`, `4`, `5`, `10` |
| Distance metric | `euclidean`, `manhattan` |
| Voting strategy | `unweighted`, `weighted` |

A total of **24 parameter combinations** are evaluated. Each configuration is assessed using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score

The best configuration is selected according to **macro F1-score**, which gives equal importance to all three target classes.

## Best Configuration

The grid search selected the following parameter combination:

| Parameter | Selected value |
|---|---|
| Distance metric | Euclidean |
| Number of neighbors | 10 |
| Voting strategy | Weighted |

The selected model therefore considers ten neighbors and assigns greater influence to neighbors that are closer to the sample being classified.

## Cross-Validated Results

Predictions for the selected pipeline are generated using stratified cross-validation. The resulting class-level performance is shown below.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Setosa | 1.000 | 1.000 | 1.000 | 50 |
| Versicolor | 0.941 | 0.960 | 0.950 | 50 |
| Virginica | 0.959 | 0.940 | 0.949 | 50 |
| **Accuracy** |  |  | **0.967** | **150** |
| **Macro average** | **0.967** | **0.967** | **0.967** | **150** |
| **Weighted average** | **0.967** | **0.967** | **0.967** | **150** |

### Aggregate Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.9667 |
| Macro precision | 0.9668 |
| Macro recall | 0.9667 |
| Macro F1-score | 0.9667 |

The selected KNN configuration correctly classified approximately **96.7%** of the cross-validated predictions. `Setosa` was classified perfectly, while the small number of errors occurred between `versicolor` and `virginica`. The nearly identical macro and weighted averages reflect the balanced class distribution and similar performance across the three species.
