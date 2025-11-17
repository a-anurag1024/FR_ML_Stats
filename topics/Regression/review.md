## ⚙️ I. Linear Regression – Foundations

### 1. **Concept Overview**

* **Goal:** Model the relationship between a dependent variable (Y) and one/more independent variables (X).
* **Equation:**
  $$(  Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \epsilon  )$$
  where $$(  \epsilon  )$$ = random error term.

### 2. **Key Assumptions**

* **Linearity:** Relationship between X and Y is linear.
* **Homoscedasticity:** Constant variance of residuals (errors) across all levels of X.
* **Exogeneity:** Independent variables are uncorrelated with the error term.
* **No Multicollinearity:** Independent variables are not highly correlated with each other.
* **Normality of Errors:** Residuals are normally distributed (important for inference).

### 3. **Diagnostic Metrics**

* **RSS (Residual Sum of Squares):**
  $$(  RSS = \sum (y_i - \hat{y_i})^2  )$$ → Measures model error. Lower is better.
* **R² (Coefficient of Determination):**
  Fraction of variance in Y explained by X.
  $$(  R^2 = 1 - \frac{RSS}{TSS}  )$$
* **Adjusted R²:**
  Adjusts R² for number of predictors. Penalizes overfitting with extra variables.
  $$(  \text{Adj }R^2 = 1 - \frac{(1 - R^2)(n-1)}{n-p-1}  )$$

### 4. **Feature Engineering & Selection**

* **Feature Selection Techniques:**

  * Forward/Backward Selection
  * Stepwise Regression
  * Statistical Significance (p-value < 0.05)
  * Information Criteria (AIC, BIC)
* **Variance Inflation Factor (VIF):**
  Detects multicollinearity.
  $$(  VIF_i = \frac{1}{1 - R_i^2}  )$$
  (VIF > 5 or 10 → high multicollinearity concern)
* **Dummy Variable Trap:**
  Occurs when all dummy variables are included → perfect multicollinearity.
  ✅ **Fix:** Drop one dummy variable (reference category).

---

## 🧮 II. Regularization in Linear Models

### 1. **Purpose:**

To reduce overfitting by penalizing large coefficients.

### 2. **Types of Regularization**


| Regularization Type | Penalty Term | Effect on Weights | Typical Use Case |
|---------------------|--------------|-------------------|------------------|
| **L1 (Lasso)** | $$( \lambda \sum_{i} mod(\beta_i) )$$ | Drives some weights exactly to **zero** → performs **feature selection** | Sparse models, high-dimensional data |
| **L2 (Ridge)** | $$( \lambda \sum_{i} \beta_i^2 )$$ | Shrinks weights smoothly toward zero but **does not** eliminate them | Preventing overfitting, multicollinearity |
| **Elastic Net** | $$( \lambda_1 \sum_i mod(\beta_i) + \lambda_2 \sum_i \beta_i^2 )$$ | Combines **sparsity** of L1 and **stability** of L2 | When variables are correlated and feature selection is desired |
| **Dropout** | Randomly drops neurons during training | Prevents co-adaptation; improves generalization | Neural networks |
| **Early Stopping** | Stops training when validation loss stops improving | Prevents overfitting without modifying the loss function | Gradient-based training, deep learning |


## 📏 III. Model Preprocessing & Validation

### 1. **Standardization**

* **Why:** Regularization and gradient descent are sensitive to feature scales.
* **How:**
  $$(  X_{scaled} = \frac{X - \mu}{\sigma}  )$$

### 2. **Cross-Validation**

* **Purpose:** Assess model generalization and prevent overfitting.
* **Common Methods:**

  * k-Fold CV (typically k=5 or 10)
  * Leave-One-Out CV
  * Stratified CV (for imbalanced classification)

---

## 🔢 IV. Logistic Regression – Classification Framework

### 1. **Concept**

* Used when dependent variable is categorical (binary).
* Models **probability** using the **logit function**:
  $$(  P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... )}}  )$$

### 2. **Cost Function**

* Derived from **Maximum Likelihood Estimation (MLE):**
  $$(  J(\beta) = -\frac{1}{N}\sum [y_i \log(\hat{y_i}) + (1 - y_i)\log(1 - \hat{y_i})]  )$$
  → Logistic loss (cross-entropy loss).

### 3. **Optimization**

* **Gradient Descent:** Iteratively minimizes cost function.
  $$(  \beta_j = \beta_j - \alpha \frac{\partial J}{\partial \beta_j}  )$$
  (α = learning rate)

---

## 📊 V. Evaluation Metrics for Classification

### 1. **Confusion Matrix**

|                     | Predicted Positive  | Predicted Negative  |
| ------------------- | ------------------- | ------------------- |
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

### 2. **Derived Metrics**

* **Accuracy:** $$(  \frac{TP + TN}{TP + TN + FP + FN}  )$$
* **Precision:** $$(  \frac{TP}{TP + FP}  )$$
* **Recall (Sensitivity / TPR):** $$(  \frac{TP}{TP + FN}  )$$
* **Specificity (TNR):** $$(  \frac{TN}{TN + FP}  )$$
* **F1-Score:** $$(  2 * \frac{Precision * Recall}{Precision + Recall}  )$$

### 3. **ROC & AUC**

* **ROC Curve:** Plots TPR vs. FPR (False Positive Rate).
* **AUC (Area Under Curve):** Measures overall separability — higher is better (1 = perfect model).

---

## 🧭 VI. Conceptual Flow Summary

| Stage                       | Concept                           | Key Goal                             |
| --------------------------- | --------------------------------- | ------------------------------------ |
| **1. Linear Regression**    | Model continuous outcomes         | Understand assumptions & diagnostics |
| **2. Regularization**       | Control overfitting               | L1/L2/ElasticNet                     |
| **3. Standardization & CV** | Improve model robustness          | Scaling + validation                 |
| **4. Logistic Regression**  | Model categorical outcomes        | Use sigmoid, optimize via GD         |
| **5. Evaluation Metrics**   | Assess classification performance | Use confusion matrix, ROC-AUC        |

---