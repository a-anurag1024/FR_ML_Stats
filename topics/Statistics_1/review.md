
---

# 📘 **Introductory Statistics**

---

## ## 1. **Classical vs Modern Statistics**

### ### A. **Population, Distribution & Classical (Frequentist) Statistics**

* Classical statistics **assumes an underlying population distribution** (often Normal), even if we only observe samples.
* The goal is to **infer population parameters** (mean, variance, proportions, etc.) using a representative sample.
* Classical methods require:

  * Assumptions about the population distribution (normality, independence, identical distribution).
  * Analytical formulas for estimators (e.g., sample mean, sample variance).
  * Closed-form confidence intervals and hypothesis tests, derived mathematically.

#### **Typical Workflow (Classical):**

1. Assume population distribution (e.g., $$(X \sim N(\mu, \sigma^2))$$).
2. Draw a sample.
3. Compute estimators (sample mean, sample variance).
4. Use known theoretical distributions (t-distribution, chi-square) to build intervals and tests.

#### **Limitations:**

* Relies heavily on assumptions.
* Not reliable when sample size is small or distribution is unknown.
* Cannot easily handle complex statistics (median, percentiles, custom estimators).

---

### ### B. **Modern Statistics (Resampling Methods — Bootstrap)**

Modern statistics removes the dependency on distributional assumptions.

#### **Bootstrap Philosophy**

* Instead of assuming a population distribution, treat the **sample itself as a proxy for the population**.
* Generate many “new samples” from the observed data **with replacement**.
* For each bootstrapped sample, compute the statistic (mean, median, regression coefficient…).
* Use the empirical distribution of these statistics to:

  * Estimate standard error.
  * Form confidence intervals.
  * Perform hypothesis testing.

#### **Advantages:**

* No need to assume normality.
* Works for complex, nonlinear statistics.
* Effective even with small samples.

#### **Limitations:**

* Assumes the original sample is representative of the population.
* Struggles with dependent data (time series) unless using block bootstrap.

---

---

## ## 2. **Sampling Theory**

### ### A. **Why Sampling Matters — Quality vs Quantity**

* A large sample is helpful **only if collected properly**.
* Sampling quality > sampling size.
* Bias destroys conclusions even if (n) is large.

**Example:** A huge dataset from a nonrepresentative survey is still biased.

---

### ### B. **Random Sampling**

* Every unit in population has equal probability of selection.
* Minimizes sampling bias.
* Forms basis for classical inference and bootstrap validity.

---

### ### C. **Stratified Sampling**

* Divide population into meaningful groups (strata), e.g., age, gender, region.
* Sample independently within each stratum.
* Improves accuracy when subpopulations differ significantly.

Benefits:

* Lower variance.
* Better representation of minority groups.
* Allows subgroup analysis.

---

### ### D. **Sampling Bias**

Occurs when some individuals in the population are systematically more likely to be sampled.

Examples:

* Internet surveys excluding older non-tech-savvy adults.
* Only interviewing people in a mall (location bias).
* Survivorship bias — only looking at successful cases.

---

### ### E. **Selection Bias**

Bias introduced due to the selection criteria used in forming the sample.

Examples:

* A medical study includes only patients who visit clinics (excludes healthy individuals).
* Self-selection in online forms.

---

### ### F. **Data Snooping / P-hacking**

* Improper reuse of the same data for testing multiple hypotheses.
* Repeatedly trying models until one shows significance.
* Leads to false discoveries.

---

### ### G. **Vast Search Effect**

* When exploring many models/hypotheses, some will appear statistically significant **just by chance**.
* Common in:

  * Big data analytics
  * High-dimensional ML
  * Genomics
  * Feature engineering with many features

Solution: Correct using techniques like Bonferroni correction, false discovery rate.

---

---

## ## 3. **Regression to the Mean**

* When an extreme observation occurs (high or low), the next observation is likely to be **closer to the average**.
* Due to natural variability, not any causal effect.

Examples:

* A student scoring exceptionally high on one test usually scores closer to average next time.
* Sports “rookie of the year” often underperform in second year.

Common Mistake: Interpreting this as a “real” improvement/decline instead of statistical tendency.

---

---

## ## 4. **Central Limit Theorem (CLT)**

### ### A. **Statement**

For independent, identically distributed random variables with finite mean and variance:

$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)
$$

As sample size grows, the **sampling distribution of the sample mean** becomes approximately normal.

---

### ### B. **Key Implications**

* You can construct confidence intervals for means even if population is NOT normal.
* Explains why normal distribution appears everywhere.

---

### ### C. **Assumptions of CLT**

1. **Independence** — observations must be independent.
2. **Identically distributed** — from same distribution.
3. **Finite mean and variance** — heavy-tailed distributions break CLT.
4. Sample size (n) should be “reasonably large”

   * Rule of thumb: $$(n \ge 30)$$ for non-heavy-tailed distributions.

---

### ### D. **Standard Error (SE)**

The standard deviation of the sampling distribution.

$$
SE = \frac{\sigma}{\sqrt{n}} \quad\text{or estimated as}\quad \frac{s}{\sqrt{n}}
$$

* Measures how much sample means vary across samples.
* Smaller SE → more precise estimate.

---

---

## ## 5. **Bootstrap Method (Modern Perspective)**

### ### A. **Purpose**

Estimate:

* Standard error
* Bias
* Confidence intervals
* Sampling distribution of any statistic

without relying on mathematical formulas or distributional assumptions.

---

### ### B. **Procedure**

1. Start with observed sample of size (n):
   ({x_1, x_2, ..., x_n})
2. Generate many bootstrap samples of size (n), sampling **with replacement**.
3. Compute statistic for each bootstrap sample.
4. Use empirical distribution of bootstrap statistics to:

   * Estimate SE
   * Construct confidence intervals

---

### ### C. **Bootstrap Confidence Intervals**

Main types:

1. **Percentile CI**
2. **Basic CI**
3. **BCa (Bias-Corrected and Accelerated)** — highest accuracy

Example (Percentile CI):

* Take 2.5th and 97.5th percentiles of bootstrap estimates → 95% CI.

---

### ### D. **Bootstrap Assumptions**

* Sample is representative of the population.
* Observations are independent.
* Sampling with replacement approximates sampling from population.
* For time-series: must use block bootstrap.

Bootstrap avoids:

* Normality assumptions
* Deriving theoretical SE
* Dependence on closed-form solutions

---

---

# ✔️ **Summary (Quick Revision Sheet)**

* Classical statistics relies on distribution assumptions; bootstrap does not.
* Good sampling is more important than large sampling.
* Biases (sampling, selection, snooping) can mislead conclusions.
* Regression to mean is a natural statistical phenomenon — not a causal effect.
* CLT allows normal approximation for means; requires iid and finite variance.
* Standard error quantifies uncertainty of sample-based estimates.
* Bootstrap provides modern, assumption-light inference tools.

---

Below are **well-structured, detailed, and cohesive review notes** covering all the distributions you listed. These are written as conceptual + interview-ready notes.

---

# 📘 **Review Notes: Common Types of Distributions in Statistics**

---

# ## **1. Normal Distribution**

### **Key Properties**

* Symmetric, bell-shaped distribution.
* Defined completely by:

  * Mean $$( \mu )$$
  * Variance $$( \sigma^2 )$$
* Many natural phenomena approximate normal distribution due to the **Central Limit Theorem**.

### **68–95–99.7 Rule**

For a normal distribution:

* **68%** of data lies within **±1 SD**
* **95%** within **±2 SD**
* **99.7%** within **±3 SD**

This helps with quick probability estimation and anomaly detection.

---

## **Q-Q Plots (Quantile–Quantile Plots)**

### **Purpose**

Graphically check whether data follows a normal distribution.

### **How Q-Q Plot Works**

* X-axis: theoretical quantiles of normal distribution
* Y-axis: sample quantiles
* If points lie approximately on a straight line → data is roughly normal.

### **Indicators of non-normality**

* **S-shaped curve** → heavy tails
* **Concave upward/downward** → skewness
* **Large deviations at ends** → outliers/heavy-tailed distribution

---

---

# ## **2. Black Swan Theory & Heavy-Tailed Distributions**

### **Black Swan Theory (Nassim Taleb)**

* Rare events with:

  1. Huge impact
  2. Extremely low probability
  3. Unpredictability
* Heavy-tailed distributions (like Cauchy, t with low df) model such rare extremal events better than normal distribution.

### **Implication**

Classical normal assumptions systematically underestimate:

* Tail risk
* Extreme losses (finance)
* Rare catastrophic failures (engineering)

---

---

# ## **3. Student’s t Distribution**

### **Definition**

Distribution of:
$$
t = \frac{\bar{X} - \mu}{s / \sqrt{n}}
$$
when population variance is unknown.

### **Key Properties**

* Symmetric like normal, but heavier tails.
* Controlled by **degrees of freedom (df)**:

  * df = (n - 1) for sample mean inference
* As **df → ∞**, t distribution → normal distribution.

---

### **Population Variance vs. Sample Variance**

* **Population variance:**
  $$(\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2)$$
* **Sample variance:**
  $$
  s^2 = \frac{1}{n - 1}\sum (x_i - \bar{x})^2
  $$
* Division by **(n-1)** introduces **unbiased estimator** → motivation for degrees of freedom.

---

### **Cauchy vs Normal Limit**

* As df ↓ (approaches 1), t distribution approaches **Cauchy distribution**.
* Cauchy:

  * No mean, no variance
  * Ultra heavy tails
  * Classical statistics fail
* As df ↑ → t becomes almost identical to normal.

---

### **Usage of t Distribution**

Used when:

* Population variance is unknown
* Sample size is small
* Constructing confidence intervals for means

Example (95% CI):
$$
\bar{x} \pm t_{\alpha/2, df} \cdot \frac{s}{\sqrt{n}}
$$

---

---

# ## **4. Binomial Distribution**

### **Definition**

Number of successes in n independent Bernoulli trials:
$$
X \sim Bin(n, p)
$$

### **Mean & Variance**

$$
E[X] = np,\quad Var[X] = np(1-p)
$$

### **Asymptotic Normality of Binomial**

For large (n), binomial approximates normal distribution:
$$
X \approx N(np, np(1-p))
$$
Works well when:

* $$(np \ge 10)$$
* $$(n(1-p) \ge 10)$$

Used in:

* Proportion tests
* Confidence intervals for population proportions

---

---

# ## **5. Chi-Square Distribution & Chi-Square Statistics**

### **Chi-Square Distribution**

If $$(Z_1, Z_2, ... Z_k)$$ are i.i.d. standard normal:
$$
\chi^2_k = \sum_{i=1}^{k} Z_i^2
$$
df = number of squared normals.

### **Pearson’s Chi-Square Statistic**

Used to compare observed vs expected frequencies:
$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

### **Uses**

1. **Goodness of Fit Test**
   Check if data follows an assumed distribution.

2. **Independence Test (Chi-Square Test for Independence)**
   Tests relationship between categorical variables.

3. **Variance Testing**
   For sample variance inference.

### **Interpretation**

Large chi-square → observed values deviate significantly → reject null.

---

---

# ## **6. F Distribution & F Statistics**

### **How F Distribution Arises**

If:

* $$(X \sim \chi^2_{d1})$$
* $$(Y \sim \chi^2_{d2})$$

Then:
$$
F = \frac{(X/d_1)}{(Y/d_2)}
$$

### **F Statistics Usage**

Measures ratio of variances:

* Typically **explained variance / unexplained variance**

---

### **In Linear Regression**

$$
F = \frac{\text{Explained variability / df}}{\text{Residual variability / df}}
$$

**Explained variance:** variability captured by the model
**Residual variance:** variability unexplained (error)

### **Use Cases**

1. **Overall significance of regression model**

   * If F is large → model explains significant variance.
2. **Comparing nested models (ANOVA)**
3. **Testing equality of two variances**

---

---

# ## **7. Poisson Distribution**

### **Definition**

Models the number of events occurring in a fixed interval (time or space):
$$
X \sim Pois(\lambda)
$$

### **Mean & Variance**

$$
E[X] = Var[X] = \lambda
$$

### **Use Cases**

* Counting rare events
* Customer arrivals per hour
* Number of defects per unit
* Call center incoming calls
* Network packet arrivals

---

---

# ## **8. Exponential Distribution**

### **Definition**

Time between two consecutive Poisson events:
$$
T \sim Exp(\lambda)
$$

### **Properties**

* Memoryless:
  $$
  P(T > s+t | T > s) = P(T > t)
  $$
* Mean = $$(1/\lambda)$$

### **Use Cases**

* Time until next:

  * phone call
  * failure
  * arrival event
* Survival modeling
* Reliability engineering

---

---

# ## **9. Weibull Distribution**

### **Definition**

$$
f(t) = \frac{k}{\lambda} \left(\frac{t}{\lambda}\right)^{k-1} e^{-(t/\lambda)^k}
$$

### **Parameters**

* **Shape parameter (k):**

  * (k < 1): decreasing hazard rate (infant mortality phase)
  * (k = 1): constant hazard (exponential distribution)
  * (k > 1): increasing hazard (wear-out failure)

* **Scale parameter (λ):**

  * Also called **characteristic life**
  * Time by which ~63.2% of population has failed

### **Probability of Failure by Time t**

$$
F(t) = 1 - e^{-(t/\lambda)^k}
$$

### **Use Cases**

* Reliability analysis
* Component lifetime modeling
* Failure rates in engineering, manufacturing
* Survival analysis

---

---

# ## **Final Summary (Snap Revision)**

| Distribution       | Key Use                                            |
| ------------------ | -------------------------------------------------- |
| **Normal**         | CLT, modeling natural phenomena, inference         |
| **t**              | CI for mean when variance unknown; heavy tails     |
| **Cauchy**         | Heavy tails & black swan modeling                  |
| **Binomial**       | Counts of successes; approx normal for large n     |
| **Chi-square**     | Goodness-of-fit, independence testing              |
| **F distribution** | Comparing variances, regression ANOVA              |
| **Poisson**        | Rare event counts                                  |
| **Exponential**    | Time between Poisson events                        |
| **Weibull**        | Failure modeling with increasing/decreasing hazard |

