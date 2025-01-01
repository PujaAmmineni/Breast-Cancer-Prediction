# Breast-Cancer-Prediction


This repository focuses on building predictive models for breast cancer diagnosis using medical imaging data. The project involves preprocessing, feature engineering, and the development of machine learning models to classify tumors as either **Benign** or **Malignant**.

## Dataset Information
- **Source**: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Sample Size**: 569
- **Predictors**:
  - **Feature Types**: Radius, Perimeter, Area, Compactness, Smoothness, Concavity, Concavity Points, Symmetry, Fractal Dimension, Texture.
  - **Statistic Types**: Mean (1), Extreme (2), Standard Error (3).
- **Response Variable**: Diagnosis (Benign or Malignant).

## Preprocessing Steps
1. **Dummy Variables**: Not applicable (no categorical variables).
2. **Handling Missing Values**: No missing values in the dataset.
3. **Handling Highly Correlated Predictors**:
   - Predictors with a correlation coefficient greater than 0.85 were removed to avoid redundancy.
   - Example: Predictors such as `concavity1`, `concave_points1`, `compactness1`, `perimeter3`, and `radius1` were removed, reducing the number of predictors from 30 to 17.
4. **Handling Uncorrelated Predictors**:
   - Predictors with low correlation to the response variable were retained as they might still contribute valuable independent information to the model.
5. **Box-Cox Transformation**: Applied to skewed variables to make their distributions more symmetric.
6. **Centering and Scaling**: Ensures predictors with varying magnitudes have equal influence.
7. **Spatial Sign Transformation**: Addresses outliers in the dataset.
8. **Data Splitting**: 80-20 stratified split for training and testing sets.

## Modeling
### Algorithms Used
- **Linear Models**: Logistic Regression, LDA, PLSDA, Penalized Models.
- **Non-linear Models**: Neural Networks, SVM, KNN, Naive Bayes, etc.

### Performance Metric
- **Primary Metric**: ROC-AUC (ideal for unbalanced binary outcomes).

### Handling Correlated and Uncorrelated Data in Modeling
- Models trained on **uncorrelated data**: Logistic Regression, LDA, QDA, RDA, MDA, FDA, KNN.
- Models trained on **correlated data**: PLSDA, Penalized Model, Neural Network, SVM, Naive Bayes.
- Using separate datasets for correlated and uncorrelated predictors allowed us to compare model performance effectively.

### Best Models
- **SVM**:
  - **Tuning Parameters**: Sigma = 0.007675, C = 4.
  - **Training ROC-AUC**: 0.9959.
  - **Testing ROC-AUC**: 1.0000.
- **Neural Network**:
  - **Tuning Parameters**: Size = 3, Decay = 1.
  - **Training ROC-AUC**: 0.9943.
  - **Testing ROC-AUC**: 0.9997.

## Results
- **Confusion Matrix for SVM**:
  | Reference   | Benign | Malignant |
  |-------------|--------|-----------|
  | **Benign**  | 70     | 0         |
  | **Malignant**| 1      | 42        |

- **Most Important Predictors**: Radius, Area, and Texture metrics.

## Key Highlights
- **Correlation Handling**: Removing highly correlated predictors improved model generalizability by reducing redundancy.
- **Dataset Separation**: Training separate models on correlated and uncorrelated datasets highlighted differences in performance across algorithms.
- SVM and Neural Networks demonstrated exceptional performance, achieving near-perfect ROC-AUC scores.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/PujaAmmineni/Breast-Cancer-Prediction.git

