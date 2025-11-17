
---

## 🌳 I. Decision Trees — Core Concepts

### 1. **Definition**

* A **non-parametric supervised learning method** used for classification and regression.
* Works by **splitting data** into subsets based on feature conditions that maximize information gain (or minimize impurity).

### 2. **Key Components**

* **Root Node:** Represents the entire dataset.
* **Internal Nodes:** Represent decision splits based on features.
* **Leaf Nodes:** Represent outcomes or predictions.

---

## 🧮 II. Splitting Criteria

### 1. **Gini Index**

* **Purpose:** Measures impurity (used in CART for classification).

* **Formula:**
  ( Gini = 1 - \sum_{i=1}^{C} p_i^2 )
  where ( p_i ) = probability of class i in a node.

* **Interpretation:**

  * Gini = 0 → pure node (all same class)
  * Gini closer to 0 → better split

**Example Calculation:**
If a node has 60% class A and 40% class B:
( Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48 )

---

### 2. **Entropy and Information Gain**

* **Entropy:** Measures uncertainty or impurity.
  ( Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i) )

* **Information Gain:** Reduction in entropy after a dataset is split on an attribute.
  ( IG = Entropy_{parent} - \sum_{k} \frac{n_k}{n} Entropy_{child_k} )

* **Interpretation:**
  Higher information gain → better feature to split.

**Example:**
Entropy before split = 0.94; after split = 0.5
→ ( IG = 0.94 - 0.5 = 0.44 )

---

## ⚙️ III. CART (Classification and Regression Trees)

* **Algorithm:** Builds a **binary tree** using **Gini index** (for classification) or **MSE** (for regression).

* **Steps:**

  1. Select the feature and threshold that best splits data (minimize Gini/MSE).
  2. Recursively repeat on child nodes.
  3. Stop when all leaves are pure or stopping criteria met.

* **Output:**

  * **Classification:** Class label at leaf.
  * **Regression:** Mean/median value at leaf.

---

## ✂️ IV. Tree Pruning (Controlling Overfitting)

### 1. **Pre-Pruning (Early Stopping)**

* Stops tree growth before it perfectly fits training data.
* **Methods:**

  * **Minimum Error / Smallest Tree Criterion:** Stop when accuracy gain < threshold.
  * **Max Depth / Min Samples Split:** Limits growth.
  * **Early Stopping:** Halt when validation performance stops improving.

### 2. **Post-Pruning**

* Grow full tree, then prune back less significant branches.
* **Approach:**

  * Evaluate nodes using validation set or cross-validation.
  * Remove splits that don’t improve accuracy.
  * Balance complexity vs. error using cost-complexity parameter (α).

---

## 🤖 V. Ensemble Learning — Overview

### 1. **Definition**

* Combines **multiple base models (weak learners)** to improve prediction accuracy, robustness, and generalization.
* **Two main goals:**

  * Reduce **variance** (via averaging/bagging)
  * Reduce **bias** (via boosting)

---

## 🧺 VI. Bagging (Bootstrap Aggregation)

### 1. **Concept**

* Trains **multiple models** on different **bootstrap samples** (random samples with replacement).
* Final prediction = **average (regression)** or **majority vote (classification)**.
* **Reduces variance** and prevents overfitting.

### 2. **Steps**

1. Generate multiple bootstrap datasets.
2. Train a model on each dataset independently.
3. Aggregate predictions.

---

## 🌲 VII. Random Forests

* **Extension of Bagging** applied to decision trees.
* **Key Differences:**

  * Uses **random subset of features** at each split (decorrelates trees).
  * **Aggregates predictions** from many trees.
* **Advantages:**

  * Reduces variance significantly.
  * Handles missing data, nonlinearities, and outliers well.
* **Hyperparameters:**

  * `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `bootstrap`.

---

## ⚡ VIII. Boosting — Reducing Bias via Sequential Learning

### 1. **Concept**

* Sequentially trains weak learners where each focuses on the **errors of previous models**.
* Learners are **weighted** by performance; final output is a weighted vote.

---

### 2. **Adaptive Boosting (AdaBoost)**

* Adjusts **weights of samples** after each round:

  * Misclassified samples get **higher weights**.
  * Correctly classified samples get **lower weights**.
* Each new weak learner focuses more on hard-to-classify points.
* **Final prediction:** Weighted sum of all weak learners.
* **Base Learner:** Shallow decision tree (stump).

---

### 3. **Gradient Boosting**

* Uses **gradient descent** to minimize loss function iteratively.
* Each new model predicts **residual errors** of the previous model.
* Works for both regression and classification.

**Steps:**

1. Start with initial prediction (mean).
2. Compute residuals (errors).
3. Fit a weak learner on residuals.
4. Update model with learning rate (shrinkage factor).
5. Repeat until convergence.

---

### 4. **XGBoost (Extreme Gradient Boosting)**

* **Optimized version of Gradient Boosting** with enhancements:

  * Regularization (L1, L2) to reduce overfitting.
  * Parallelized tree construction.
  * Tree pruning via “max_depth”.
  * Shrinkage and column subsampling for speed and robustness.
  * Handles missing data internally.

* **Advantages:**

  * Fast, scalable, regularized, and interpretable.
  * Dominant in ML competitions (e.g., Kaggle).

---

## 🧠 IX. Stacking (Stacked Generalization)

### 1. **Concept**

* Combines **different types of models (heterogeneous)** rather than identical ones.
* **Meta-learner:** A model trained on the predictions of base learners.
* **Process:**

  1. Train base models (e.g., Decision Tree, SVM, Logistic Regression).
  2. Collect predictions on validation set.
  3. Train meta-model on these predictions to output final result.

---

## 🧩 X. Summary Table

| Concept                  | Goal                      | Key Method                | Effect                |
| ------------------------ | ------------------------- | ------------------------- | --------------------- |
| **Gini Index / Entropy** | Measure impurity          | CART, ID3, C4.5           | Choose best split     |
| **CART**                 | Tree-building algorithm   | Gini/MSE                  | Binary tree           |
| **Pre-Pruning**          | Prevent overfitting early | Depth, min_samples        | Simplify tree         |
| **Post-Pruning**         | Prune after full tree     | Cost-complexity           | Better generalization |
| **Bagging**              | Reduce variance           | Bootstrap + aggregation   | Random Forest         |
| **Boosting**             | Reduce bias               | Sequential correction     | AdaBoost, GBM         |
| **XGBoost**              | Optimized boosting        | Gradient + Regularization | Fast, scalable        |
| **Stacking**             | Combine diverse models    | Meta-learner              | Enhanced accuracy     |

---