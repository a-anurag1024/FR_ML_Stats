
---

# 📘 **Linear Regression — Detailed Review Notes**

---

# **1. Introduction to Linear Regression**

Linear regression models the relationship between a **dependent variable (y)** and one or more **independent variables (X)** using a linear function.

### **Model Form**

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon
$$

* $$(\beta_i)$$: parameters (coefficients)
* $$(\epsilon)$$: error term assumed to be normally distributed with mean 0 and variance ($$\sigma^2$$)

---

# **2. Ordinary Least Squares (OLS)**

OLS is the most common method to estimate regression parameters.

### **Objective: Minimize RSS**

**Residual Sum of Squares**:
$$
RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### **Closed-Form Solution**

Given (X) as an $$(n \times p)$$ matrix and (y) as a vector:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

Interpretation:

* Choose coefficients that minimize squared residuals.
* Geometric viewpoint: project (y) onto the column space of (X).

### **When does OLS work poorly?**

* High multicollinearity
* p > n
* Outliers
* Non-linearity
* Heteroskedasticity

---

# **3. Evaluating Linear Regression Fit**

## **3.1 Root Mean Squared Error (RMSE)**

$$
RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}
$$

* Measures average prediction error.
* In original scale of target variable.

---

## **3.2 Residual Standard Error (RSE)**

$$
RSE = \sqrt{\frac{RSS}{n - p - 1}}
$$

* Estimates the standard deviation of the errors.
* Penalizes model for using many predictors.

---

## **3.3 R² (Coefficient of Determination)**

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

Where (TSS) = total sum of squares.

Interpretation:

* Fraction of variance explained by the model.
* R² always increases when adding predictors (even useless ones).

---

## **3.4 Adjusted R²**

$$
\text{Adjusted }R^2 = 1 - \frac{(RSS/(n - p - 1))}{(TSS/(n - 1))}
$$

* Penalizes unnecessary predictors.
* Increases only if new predictor improves model more than expected by chance.

---

# **4. Significance of Coefficients: t-statistics & p-values**

OLS estimates come with standard errors.
Each coefficient has a **hypothesis test**:

### **Null Hypothesis**

$$
H_0: \beta_j = 0
$$

### **t-statistic**

$$
t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
$$

### **p-value**

Probability of observing such a t-statistic assuming $$(H_0)$$ is true.

### **Interpretation**

* Small p-value → variable contributes significantly.
* Large p-value → coefficient may be noise → candidate for removal.

### **Feature selection via significance**

* Start with full model → remove high p-value predictors.
* Used often with backward selection.

---

# **5. Penalizing Model Complexity: AIC, BIC, AICc, Mallows’ Cp**

When selecting a model, balancing fit vs complexity is essential.

---

## **5.1 Akaike Information Criterion (AIC)**

$$
AIC = 2p + n \ln(RSS/n)
$$

* Penalizes number of parameters.
* Lower AIC = better model.

---

## **5.2 Bayesian Information Criterion (BIC)**

$$
BIC = p \ln(n) + n \ln(RSS/n)
$$

* Larger penalty for complexity than AIC.
* More conservative.

---

## **5.3 AICc (Corrected AIC)**

$$
AICc = AIC + \frac{2p(p+1)}{n - p - 1}
$$

* Corrects AIC for small samples.
* Important when (n) is not much larger than (p).

---

## **5.4 Mallows’ Cp**

$$
C_p = \frac{RSS_p}{\hat{\sigma}^2} - (n - 2p)
$$

* Compares model with (p) predictors to full model.
* Good model → $$(C_p)$$ close to (p).

---

# **6. Feature Selection Methods**

### **6.1 Forward Selection**

* Start with no predictors.
* Add predictors one by one (based on AIC/BIC/p-value).

### **6.2 Backward Selection**

* Start with all predictors.
* Remove predictors one by one (based on p-value, BIC).

### **6.3 Stepwise Selection**

* Combination of forward and backward selection.

---

## **6.4 L1 and L2 Regularization**

### **L1 Regularization (Lasso)**

$$
\text{Minimize } RSS + \lambda \sum |\beta_j|
$$

Effects:

* Shrinks coefficients.
* Performs feature selection by setting some coefficients to zero.

---

### **L2 Regularization (Ridge)**

$$
\text{Minimize } RSS + \lambda \sum \beta_j^2
$$

Effects:

* Shrinks coefficients but does not eliminate any.
* Useful for multicollinearity.

---

### **Elastic Net**

Combination of L1 + L2.

---

# **7. Confidence Intervals & Prediction Intervals**

### **7.1 Confidence Interval for Mean Prediction**

Range of plausible values for the **mean response** at a given X.

$$
\hat{y} \pm t_{\alpha/2, df} \cdot SE(\hat{y})
$$

---

### **7.2 Prediction Interval**

Range for **individual predicted observation**.

$$
\hat{y} \pm t_{\alpha/2, df} \cdot \sqrt{SE(\hat{y})^2 + \sigma^2}
$$

* Wider than confidence intervals.
* Includes irreducible error.

---

## **7.3 Using Resampling to Estimate Confidence/Prediction Intervals**

### **Bootstrap Method**

1. Resample dataset with replacement.
2. Fit regression model.
3. Compute predictions.
4. Use quantiles of bootstrap distribution for confidence/prediction intervals.

Advantages:

* Does not rely on normality assumptions.
* Works well with small samples.

---

# **8. Factor Variables and Encoding**

Categorical (factor) variables need conversion to numeric form.

---

## **8.1 One-Hot Encoding**

* Convert each category → binary indicator variable.
* If k levels → k−1 dummy variables (to avoid dummy trap).
* Reference category absorbed into intercept.

---

## **8.2 Factor Variable with Multiple Levels**

Example:

* Region = {North, South, East, West}
* Model includes dummy variables to represent each region relative to baseline.

---

## **8.3 Ordered Factor Variables**

Categories have natural order:

* e.g., education level, satisfaction level.

Two approaches:

### **(a) Encode as integers (ordinal encoding)**

Preserves order but imposes linearity.

### **(b) One-hot encoding + monotonic constraints**

More flexible but more parameters.

---

# **9. Summary Table**

| Concept          | Key Idea                                     |
| ---------------- | -------------------------------------------- |
| OLS              | Minimizes RSS to find best-fit line          |
| RMSE             | Measures average prediction error            |
| RSE              | Estimates standard deviation of residuals    |
| R²               | Percentage variance explained                |
| Adj. R²          | Penalizes extra predictors                   |
| t-statistic      | Determines significance of coefficients      |
| p-value          | Probability of coefficient arising by chance |
| AIC/BIC          | Penalize model complexity                    |
| L1/L2            | Shrink or select features                    |
| CIs              | Uncertainty for mean predictions             |
| PIs              | Uncertainty for individual predictions       |
| Bootstrapping    | Non-parametric CI/PI estimation              |
| Factor variables | Encoded using dummy variables                |
| Ordered factors  | Encode ordering carefully                    |

---

Below is a **well-structured, detailed review** on **Regression Diagnostics**, covering all the topics you listed.
This is formatted as clean study notes suitable for a Jupyter notebook or interview prep.

---

# 📘 **Regression Diagnostics**

Regression diagnostics help validate assumptions of linear regression, identify problematic patterns or data points, and guide model improvements.

---

# **1. Multicollinearity**

Multicollinearity occurs when two or more predictors are **highly correlated**, making it difficult to interpret coefficients.

### **Problems Caused by Multicollinearity**

* Inflated standard errors → unstable coefficients
* t-values shrink → insignificant predictors despite strong relationship
* Coefficient signs may flip
* Small changes in data lead to large coefficient changes

### **Detection Methods**

* **Correlation matrix**
* **Variance Inflation Factor (VIF)**:
  $$
  VIF = \frac{1}{1 - R_j^2}
  $$
  Rule: VIF > 5 or 10 indicates serious multicollinearity.
* **Condition Number (κ)** of X matrix

### **Fixes**

* Remove redundant predictors
* Combine correlated predictors
* Use PCA or dimensionality reduction
* Apply regularization (Ridge handles multicollinearity well)

---

# **2. Confounding Variables**

A confounder is a variable that:

1. Influences both the predictor and the outcome
2. Creates a false or exaggerated association between them

### **Example**

Predictor: number of firefighters
Outcome: fire damage
Confounder: fire intensity

### **Effects**

* Bias in coefficient estimates
* Incorrect causal interpretation
* Spurious correlations

### **Fixes**

* Include confounders in the model
* Use stratification
* Randomization in experimental design
* Propensity scoring in observational data

---

# **3. Main Effects & Interaction Effects**

## **3.1 Main Effect**

Effect of a predictor on the response **holding all other variables constant**.

Example:
Effect of dosage on blood pressure reduction.

---

## **3.2 Interaction Effect**

Occurs when the effect of one predictor **depends on the level of another predictor**.

### **Interaction Term in Regression**

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 x_2) + \epsilon
$$

* $$(x_1 x_2)$$: **interaction term**
* $$(\beta_3)$$: quantifies how relationship between $$(x_1)$$ and $$(y)$$ changes with $$(x_2)$$

### **Interpretation**

* If $$(\beta_3 \neq 0)$$, the effect of $$(x_1)$$ is modified by $$(x_2)$$
* Interaction often more important than main effects

---

# **4. Outliers & Standardized Residuals**

### **Outlier Definition**

Point that deviates strongly from trend of other data.

### **Standardized Residuals**

Residual divided by its estimated standard deviation:
$$
e_i^{*} = \frac{e_i}{\hat{\sigma} \sqrt{1 - h_i}}
$$

Where $$(h_i)$$ is leverage (row influence).
Rule of thumb:

* |standardized residual| > 2 or 3 indicates potential outlier.

### **Use Case in Anomaly Detection**

* Points with high residuals represent **unexpected patterns**.
* Useful for identifying faulty sensors, payment fraud, broken process measurements.

---

# **5. Influential Values (High-Influence Points)**

Outliers ≠ influential points.
An **influential point** is one that significantly changes regression estimates.

### **Measures to Detect Influential Points**

#### **(a) Leverage $$(h_i)$$**

* Measures how unusual a predictor vector is
* High leverage means predictor is far from mean

Rule:
$$
h_i > \frac{2p}{n} \text{ (or } \frac{3p}{n})
$$

---

#### **(b) Cook’s Distance (Dₖ)**

$$
D_i = \frac{r_i^2}{p , \hat{\sigma}^2} \cdot \frac{h_i}{(1-h_i)^2}
$$

* Measures influence on all fitted values
* D > 1 often indicates influential point

---

#### **(c) DFBETAS**

Measures influence of a point on a single coefficient.

### **Summary**

* Outlier: unusual **y** value
* High leverage: unusual **X** value
* Influential: both large leverage + affects coefficients

---

# **6. Heteroskedasticity**

Violation of assumption:
$$
Var(\epsilon_i) = \sigma^2 \quad \text{constant}
$$

### **Symptoms**

* Residuals fan out or funnel in
* Variance depends on predictor

### **Problems**

* OLS coefficients still unbiased
* Standard errors wrong → invalid **t-tests & CIs**
* Incorrect inference (false significance)

### **Detection**

* Residuals vs fitted plot
* Breusch–Pagan test
* White test

### **Fixes**

* Transform response (log, square root)
* Weighted least squares
* Robust standard errors (“sandwich” estimators)

---

# **7. Partial Residual Plots**

Partial residual plots show **relationship between a predictor and response**, controlling for other predictors.

### **Partial Residual**

$$
\text{Partial residual} = r_i + \hat{\beta}*j x*{ij}
$$

Where:

* $$(r_i = y_i - \hat{y}_i)$$
* For predictor $$(x_j)$$

### **Purpose**

* Detect **nonlinearity** of a predictor
* Reveal if transformations (log, polynomial) are needed
* Check effect masked by other predictors

### **Indicates Nonlinearity When**

* Pattern not linear
* Curvature visible in partial residuals

---

# **8. Polynomial Regression**

A way to model nonlinear relationships while staying within linear modeling framework.

### **Model**

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_d x^d + \epsilon
$$

### **Notes**

* Still linear in parameters ($$(\beta)$$), so OLS applies
* Risk of overfitting with high degree
* Often used after detecting nonlinearity via diagnostics

### **Selection of degree**

* Use CV, AIC, BIC
* Visual inspection via partial residual plots

---

# **9. Splines & Piecewise Polynomials**

Splines generalize polynomial regression by fitting **piecewise polynomials** that join smoothly.

### **Definitions**

#### **(a) Piecewise Polynomial**

Different polynomial in each region of X.

#### **(b) Knots**

Points where polynomial pieces meet
(e.g., knot at x = 5 means polynomial changes form at 5)

---

## **9.1 Regression Splines**

* Fit low-degree polynomials within intervals
* Ensure continuity and smoothness at knots
* More stable and flexible than high-degree polynomials

### **Types of Splines**

1. **Linear Splines** (piecewise linear)
2. **Quadratic or Cubic Splines**
3. **Restricted cubic splines (natural splines)**

   * Constrained to be linear beyond boundaries

### **Advantages of Splines**

* Capture complex nonlinear relationships
* Avoid global polynomials’ wild oscillations
* Smooth and interpretable

### **Choosing Number/Location of Knots**

* Domain knowledge
* Cross-validation
* Quantile-based placement (e.g., knots at 25/50/75 percentiles)

---

# **10. Summary Table**

| Topic                  | Key Idea                                                   |
| ---------------------- | ---------------------------------------------------------- |
| Multicollinearity      | Predictors strongly correlated → unstable coefficients     |
| Confounding            | Hidden variable affects both X and Y → biased coefficients |
| Interaction Term       | Effect of X₁ depends on level of X₂                        |
| Standardized Residuals | Measure outliers; used in anomaly detection                |
| Influential Points     | Leverage + impact on coefficients; detected via Cook’s D   |
| Heteroskedasticity     | Non-constant variance; invalidates standard inference      |
| Partial Residual Plots | Diagnose nonlinearity for individual predictor             |
| Polynomial Regression  | Models curvature using higher-order terms                  |
| Splines                | Flexible piecewise polynomials with knots                  |


