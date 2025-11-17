
---

## 🧭 I. K-Nearest Neighbors (K-NN)

### 1. **Concept Overview**

* A **non-parametric, instance-based** learning algorithm.
* Predicts output based on **majority class (classification)** or **average value (regression)** of the K nearest data points.
* **Key idea:** “Objects close to each other are likely to be similar.”

### 2. **Distance Measures**

| Metric                 | Formula                           | Use Case                |             |                                                        |
| ---------------------- | --------------------------------- | ----------------------- | ----------- | ------------------------------------------------------ |
| **Euclidean Distance** | $$(  d = \sqrt{\sum (x_i - y_i)^2}  )$$ | Continuous numeric data |             |                                                        |
| **Manhattan Distance** | $$(  d = \sum mod(x_i - y_i)  )$$           | Grid-like structures (e.g., city blocks)               |
| **Minkowski Distance** | $$(  d = (\sum  (x_i - y_i)^p)^{1/p}  )$$ | Generalized version (p=1 → Manhattan, p=2 → Euclidean) |
| **Hamming Distance**   | Count of mismatched bits          | Categorical/binary data |             |                                                        |

### 3. **Optimized KNN**

* **Challenge:** High computation cost for large datasets.
* **Optimizations:**

  * Use **KD-Trees** (low dimensions) or **Ball Trees** (high dimensions).
  * **Hyper-sphere search:** Reduces unnecessary distance computations using geometric boundaries.
  * **Box-shaped hyperplanes:** Partitions space into axis-aligned boxes for faster nearest-neighbor lookup.

### 4. **Choosing K**

* Small K → noisy & overfitting
* Large K → oversmoothing
* Use **cross-validation** to select optimal K.

---

## ⚔️ II. Support Vector Machines (SVM)

### 1. **Concept**

* A **supervised learning** algorithm for classification/regression.
* Finds an **optimal hyperplane** that **maximizes the margin** between data points of different classes.
* **Support vectors** are the data points closest to the decision boundary.

### 2. **Mathematical Objective**

* Maximize margin:
  $$(  \frac{2}{||w||}  )$$
  Subject to: $$(  y_i(w \cdot x_i + b) \ge 1  )$$

### 3. **Kernel Trick**

* **Purpose:** Transform non-linear data into higher-dimensional space where it becomes linearly separable.
* **Common Kernels:**

  * **Linear:** $$(  K(x, y) = x^T y  )$$
  * **Polynomial:** $$(  K(x, y) = (x^T y + c)^d  )$$
  * **RBF (Gaussian):** $$(  K(x, y) = \exp(-\gamma ||x - y||^2)  )$$
  * **Sigmoid:** $$(  K(x, y) = \tanh(\alpha x^T y + c)  )$$

---

## 🧩 III. Semi-Supervised Learning

* **Definition:** Combines **labeled + unlabeled** data for training.
* Useful when labeling is expensive or time-consuming.
* **Examples:** Self-training, Label Propagation, Co-training.
* **Applications:** Web classification, medical image labeling, speech recognition.

---

## 🎯 IV. K-Means Clustering

### 1. **Concept**

* An **unsupervised** algorithm that partitions data into **K clusters** by minimizing **intra-cluster variance**.
* Each point belongs to the cluster with the nearest mean (centroid).

### 2. **Objective Function**

$$(  J = \sum_{i=1}^{K}\sum_{x \in C_i} ||x - \mu_i||^2  )$$

### 3. **Elbow Method**

* Used to find optimal **K**.
* Plot WCSS (Within-Cluster Sum of Squares) vs. K → the “elbow point” indicates diminishing returns.

### 4. **WCSS (Within Cluster Sum of Squares)**

* Measures compactness of clusters:
  $$(  WCSS = \sum_{i=1}^{K}\sum_{x \in C_i} ||x - \mu_i||^2  )$$

### 5. **K-Means++ Initialization**

* Improves centroid initialization:

  1. Choose one random centroid.
  2. Select next centroid with probability ∝ distance² from nearest chosen centroid.
  3. Repeat until K centroids chosen.

### 6. **Problems with K-Means**

* Sensitive to initial centroids.
* Requires predefined K.
* Assumes spherical clusters, equal variance.
* Struggles with **non-linear boundaries** and **outliers**.

### 7. **Mini-Batch K-Means**

* Processes **small random batches** instead of the whole dataset for faster convergence on large data.
* Reduces computation cost with minimal accuracy loss.

---

## 🧬 V. Hierarchical Clustering

### 1. **Overview**

* Builds hierarchy of clusters in a **tree-like structure (dendrogram)**.
* **Two types:**

  * **Agglomerative (Bottom-Up):** Start with individual points and merge clusters iteratively.
  * **Divisive (Top-Down):** Start with one cluster and recursively split.

### 2. **Linkage Methods**

| Linkage Type         | Formula / Concept                | Description                                                 |
| -------------------- | -------------------------------- | ----------------------------------------------------------- |
| **Single Linkage**   | min(distance between points)     | Sensitive to noise (chaining effect).                       |
| **Complete Linkage** | max(distance between points)     | Produces compact, spherical clusters.                       |
| **Average Linkage**  | mean(distance between all pairs) | Balance between single and complete.                        |
| **Centroid Linkage** | distance between centroids       | Can lead to “inversion” (clusters merging then separating). |

---

## 🌐 VI. Density-Based Clustering (DBSCAN)

### 1. **Concept**

* Groups data points **based on density** — areas with high density separated by low-density regions.
* Handles **non-spherical clusters** and **noise** effectively.

### 2. **Parameters**

* **ε (epsilon):** Radius of neighborhood.
* **MinPts:** Minimum number of points required to form a dense region.

### 3. **Point Categories**

* **Core Point:** ≥ MinPts within ε.
* **Border Point:** < MinPts but within ε of a core point.
* **Noise Point:** Neither core nor border.

### 4. **Advantages**

* No need to specify K.
* Handles arbitrary-shaped clusters.
* Robust to outliers.

### 5. **Disadvantages**

* Sensitive to ε and MinPts.
* Struggles in varying density datasets.

---

## 📊 VII. Cluster Evaluation Metrics

| Metric               | Type     | Description                                                                                            |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------------ |
| **Silhouette Score** | Internal | Measures how similar an object is to its own cluster vs others. Ranges from -1 to 1 (higher = better). |
| **Gold Standard**    | External | Ground-truth-based comparison of clustering with true labels.                                          |
| **RAND Index**       | External | Compares pairs of points for agreement in clustering vs true labels.                                   |
| **Jaccard Index**    | External | Ratio of true positives to union of true/false positives.                                              |
| **Purity**           | External | Fraction of dominant class in each cluster (simple interpretability).                                  |

---

## 🧠 VIII. Principal Component Analysis (PCA)

### 1. **Concept**

* **Dimensionality reduction** technique projecting data into directions (components) of **maximum variance**.
* Converts correlated features into uncorrelated principal components.

### 2. **Mathematics**

1. Standardize data.
2. Compute covariance matrix $$(  Σ = \frac{1}{n-1}X^T X  )$$.
3. Calculate eigenvalues & eigenvectors.
4. Sort eigenvectors by descending eigenvalues.
5. Select top-k components.

### 3. **Interpretation**

* **First PC:** Direction of max variance.
* **Explained Variance Ratio:** Measures information retained after dimensionality reduction.
* **Use cases:** Noise reduction, visualization, speed-up for downstream ML tasks.

---

## 🧩 IX. Summary Table

| Technique           | Learning Type                | Key Idea                              | Strength                        | Limitation                  |
| ------------------- | ---------------------------- | ------------------------------------- | ------------------------------- | --------------------------- |
| **K-NN**            | Supervised                   | Majority vote among nearest neighbors | Simple, non-parametric          | Slow for large data         |
| **SVM**             | Supervised                   | Maximize margin between classes       | Works well in high dimensions   | Computationally heavy       |
| **Semi-Supervised** | Mixed                        | Combines labeled + unlabeled data     | Leverages unlabeled data        | Hard to tune assumptions    |
| **K-Means**         | Unsupervised                 | Partition data into K clusters        | Simple, fast                    | Sensitive to initialization |
| **Hierarchical**    | Unsupervised                 | Nested clusters via linkage           | No need for K                   | Expensive for large data    |
| **DBSCAN**          | Unsupervised                 | Density-based clustering              | Handles noise, arbitrary shapes | Fails with varying density  |
| **PCA**             | Unsupervised (preprocessing) | Project onto max variance axes        | Reduces dimensionality          | Loses interpretability      |

---