# Bank Marketing Classification

## Decision Tree and Random Forest Analysis

This project applies tree-based machine learning methods to the **Bank Marketing dataset** to predict whether a customer will subscribe to a term deposit. It evaluates an initial Decision Tree classifier, investigates the effect of selected Decision Tree hyperparameters, and compares the results with a Random Forest classifier.

## Problem Statement

The Bank Marketing dataset contains records of direct marketing contacts made by a banking institution. Each row represents an interaction with a customer and includes customer-related and campaign-related attributes.

The target variable is `deposit`, which indicates whether the customer subscribed to a term deposit:

- `yes` — the customer subscribed
- `no` — the customer did not subscribe

The objective is to build a binary classification model that predicts the subscription outcome and to evaluate its performance using multiple classification metrics.

## Project Objectives

- Prepare the Bank Marketing dataset for machine learning
- Train and evaluate a Decision Tree classifier
- Measure performance using accuracy, precision, recall, and a classification report
- Examine how `max_depth` and `min_samples_split` affect Decision Tree performance
- Train a Random Forest classifier with 100 estimators
- Compare the predictive performance of the evaluated models
- Record the Random Forest training time

## Dataset Preparation

The following preprocessing steps are applied:

1. The dataset is loaded into a Pandas DataFrame.
2. The target column, `deposit`, is mapped from `yes` and `no` to `1` and `0`.
3. The remaining categorical columns are converted to numerical values using `LabelEncoder`.
4. The features and target are separated.
5. The data is divided into training and test sets using `train_test_split` with `random_state=256`.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** the proportion of all predictions that are correct
- **Precision:** the proportion of predicted subscribers who actually subscribed
- **Recall:** the proportion of actual subscribers correctly identified by the model
- **F1-score:** the harmonic mean of precision and recall, reported for the initial Decision Tree

## Experiments and Results

### 1. Initial Decision Tree

An unconstrained `DecisionTreeClassifier` is trained with `random_state=50` to establish the initial model performance.

| Metric | Score |
|---|---:|
| Accuracy | 0.7800 |
| Precision | 0.7689 |
| Recall | 0.7666 |

The initial Decision Tree achieved an accuracy of approximately **78.0%** on the test set.

#### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 — No subscription | 0.79 | 0.79 | 0.79 | 1,467 |
| 1 — Subscription | 0.77 | 0.77 | 0.77 | 1,324 |
| **Accuracy** |  |  | **0.78** | **2,791** |
| **Macro average** | **0.78** | **0.78** | **0.78** | **2,791** |
| **Weighted average** | **0.78** | **0.78** | **0.78** | **2,791** |

The similar macro and weighted averages indicate relatively balanced performance across the two target classes in this test set.

### 2. Decision Tree Hyperparameter Analysis

Two additional Decision Tree configurations are evaluated using different values for:

- `max_depth`: the maximum depth of the tree
- `min_samples_split`: the minimum proportion of samples required to split an internal node

| Configuration | `max_depth` | `min_samples_split` | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Decision Tree A | 5 | 0.1 | 0.7753 | 0.7442 | 0.8021 |
| Decision Tree B | 15 | 0.3 | 0.7499 | 0.7001 | 0.8270 |

The first configuration retained accuracy close to the initial Decision Tree while increasing recall to approximately **80.2%**. The second configuration produced the highest Decision Tree recall, approximately **82.7%**, but its precision and accuracy decreased.

These results demonstrate a clear trade-off: the evaluated parameter changes allowed the model to identify a larger proportion of actual subscribers, but also increased the number of false-positive predictions.

### 3. Random Forest

A `RandomForestClassifier` with 100 estimators and `random_state=55` is trained on the same training data.

| Metric | Result |
|---|---:|
| Accuracy | 0.8456 |
| Precision | 0.8169 |
| Recall | 0.8693 |
| Training time | 1.53 seconds |

The Random Forest achieved approximately **84.6% accuracy**, **81.7% precision**, and **86.9% recall**. The measured training time was approximately **1.53 seconds** in the recorded execution environment.

## Model Comparison

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Initial Decision Tree | 0.7800 | 0.7689 | 0.7666 |
| Decision Tree A | 0.7753 | 0.7442 | 0.8021 |
| Decision Tree B | 0.7499 | 0.7001 | 0.8270 |
| **Random Forest** | **0.8456** | **0.8169** | **0.8693** |

Among the evaluated models, the Random Forest produced the highest accuracy, precision, and recall. Its accuracy was approximately **6.6 percentage points** higher than that of the initial Decision Tree.
