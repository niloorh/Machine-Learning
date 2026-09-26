# Machine Learning Coursework

This repository contains four machine learning assignments developed in Python and Jupyter Notebook. Together, they cover a broad learning path from classical supervised models and regression to deep neural networks and unsupervised clustering.

The assignments focus on practical model development: preparing data, implementing algorithms, testing different configurations, selecting suitable evaluation metrics, and interpreting both numerical and visual results.

## Assignment Overview

| Assignment | Main focus | Methods and concepts |
|---|---|---|
| **01 — Supervised Classification** | Classification with classical machine learning models and comparison of different decision strategies | Decision Trees, Random Forests, custom KNN, distance metrics, weighted voting, hyperparameter tuning, cross-validation |
| **02 — Linear Models** | Continuous-value prediction and binary classification under class imbalance | Linear Regression, custom Logistic Regression, gradient descent, SGD, feature encoding, oversampling, class weighting |
| **03 — Neural Networks** | Image classification with fully connected, convolutional, residual, and pretrained networks | MLP, CNN, ResNet, data augmentation, dropout, L2 regularization, batch normalization, transfer learning, fine-tuning |
| **04 — Clustering Algorithms** | Discovering behavioral groups without using the original class labels | K-Means, DBSCAN, Elbow Method, silhouette score, outlier detection, PCA visualization |

## Work Covered

The assignments include regression, binary classification, multiclass classification, image recognition, and clustering. Both custom implementations and established machine learning libraries are used, allowing the internal behavior of selected algorithms to be studied alongside more advanced models.

Data preparation varies across the projects and includes:

- Handling missing values
- Encoding categorical and binary variables
- Standardizing numerical features
- Managing imbalanced classes
- Applying image normalization and data augmentation

Model development includes manual algorithm implementation, parameter search, cross-validation, optimizer comparison, regularization, residual learning, and transfer learning. The experiments also examine how different modeling choices affect convergence, generalization, and class-level performance.

## Evaluation and Analysis

| Problem type | Evaluation methods |
|---|---|
| Regression | Mean Squared Error, Root Mean Squared Error, Mean Absolute Error |
| Classification | Accuracy, precision, recall, F1-score, classification reports, confusion matrices |
| Clustering | Within-cluster sum of squares, silhouette score, cluster count, outlier count |
| Neural networks | Training and validation curves, test loss, test accuracy, optimizer and architecture comparison |

Visual analysis is used throughout the assignments to make model behavior easier to interpret. The included outputs cover learning curves, confusion matrices, coefficient evolution, clustering projections, and comparisons between alternative model configurations.

Overall, the repository demonstrates a progression from fundamental machine learning techniques to deeper architectures and unsupervised analysis, with an emphasis on appropriate preprocessing, fair model comparison, and careful interpretation of results.

