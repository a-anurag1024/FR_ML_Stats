
---

# **📘 Introduction to Hypothesis Testing — Detailed Review Notes**

---

## **1. What is Hypothesis Testing?**

Hypothesis testing is a statistical framework used to decide whether there is enough evidence in a sample to infer that a certain condition is true for the entire population.

The idea:

* Assume a claim (null hypothesis).
* Use data to test the likelihood of observing what we saw *if the claim were true*.
* Based on that likelihood, decide whether to reject the claim.

---

# **2. Null vs. Alternative Hypothesis**

### **Null Hypothesis (H₀)**

* The default assumption: *no effect*, *no difference*, or *status quo*.
* Examples:

  * "The mean conversion rate in A and B are equal."
  * "A new drug has no effect beyond placebo."

### **Alternative Hypothesis (H₁ or Hₐ)**

* Competing claim that *there is an effect* or *there is a difference*.
* Examples:

  * "Conversion rate of B is higher than A."
  * "Drug has an effect."

---

# **3. One-Tailed vs. Two-Tailed Tests**

### **One-Tailed Test**

Checks if the effect is in a specific direction.

* H₁: μ₁ > μ₂ (or <)

Example:
“We expect version B to increase conversions.”

### **Two-Tailed Test**

Checks for any difference, irrespective of direction.

* H₁: μ₁ ≠ μ₂

Example:
“We just want to know if there is *any* difference between A and B.”

**Choice depends on the scientific/business question** — two-tailed is more conservative.

---

# **4. Hypothesis Testing Use-Cases**

### **A/B Testing**

* Comparing two versions (A vs B)
* Area: product changes, UI experiments, ads, email subject lines

### **Control vs. Treatment Groups**

* Control: baseline condition
* Treatment: receives the intervention
* Used in medical trials, marketing, recommendation systems

### **Quality Control**

* Assess if manufacturing meets standards

### **Scientific Research**

* Testing difference between groups, associations, effects

### **Business Analytics**

* Customer churn, impact of promotions, funnel analysis

---

# **5. Resampling Approaches to Hypothesis Testing**

Resampling tests rely on repeated random sampling from the observed data rather than theoretical distributions.

Why use resampling?

* Distribution-free (non-parametric)
* Works with small samples
* Useful when normality assumptions fail

Common Resampling Techniques:

1. **Permutation Tests**
2. **Bootstrap-based Hypothesis Tests**

---

# **6. Permutation Tests**

A permutation test (randomization test) checks whether two samples are different by repeatedly shuffling labels and computing the “test statistic” under random assignment.

### **How It Works**

1. Compute an observed statistic (e.g., difference in means).
2. Combine samples → repeatedly shuffle labels → split them → compute statistic each time.
3. Compare observed value to distribution of permuted values.

### **When Useful?**

* Non-normal data
* Small sample sizes
* A/B tests
* Any two-sample comparison

---

## **Variants of Permutation Tests**

### **(a) Exact Permutation Test**

* Considers *all possible* permutations of labels.
* Gives exact p-value.
* Feasible only for small datasets.

### **(b) Approximate / Monte-Carlo Permutation Test**

* Randomly samples permutations instead of enumerating all.
* Practical for large datasets.

### **(c) Bootstrap Resampling Test**

Bootstrap ≠ permutation (but related conceptually).

**Bootstrap test approach:**

1. Resample *with replacement* from each sample separately.
2. Generate bootstrap samples under H₀ by adjusting samples to have same mean (or another constraint).
3. Compute distribution of the statistic.
4. Compare observed statistic.

**Used when:**

* Want confidence interval + hypothesis test
* Want to avoid assumptions about sampling distribution
* Two samples have unequal variances or distributions

---

# **7. Understanding p-Value**

### **Common Misinterpretation:**

❌ *“Probability that the null hypothesis is true.”*
This is **incorrect**.

### **Correct Interpretation:**

✔ **Probability of observing our data (or more extreme) *assuming the null is true*.**

### **Key Idea:**

p-value tells us **how surprising the data is under H₀**, not the probability of H₀.

---

## **Limitations of p-Values**

* Highly sensitive to sample size (with huge N, everything becomes “significant”).
* Doesn’t tell about effect size.
* Doesn’t give probability of hypotheses.
* p-hacking possible (manipulate analyses until p < 0.05).

---

## **p-Hacking**

Bad statistical practice—researchers repeatedly:

* Try multiple models
* Try multiple variables
* Try many subgroup analyses
* Drop variables
* Stop data collection early
* Choose that one analysis where p < 0.05

This inflates false positives (Type I errors).

---

# **8. Effect Size**

p-value tells **IF** an effect exists.
Effect size tells **HOW MUCH** effect exists.

Examples of effect-size metrics:

* Cohen’s d (difference between means)
* Pearson’s r
* Risk ratios / Odds ratios
* Percentage lift in A/B testing

Importance: Large studies can detect tiny effects that do not matter.
Effect size captures practical significance.

---

# **9. Alpha Value (Significance Level α)**

* Predefined threshold for rejecting H₀.
* Common choice: **α = 0.05** (5% significance level).
* Represents allowed Type I error rate.

Meaning:
If α = 0.05, you accept a 5% chance of falsely rejecting a true null.

---

# **10. Type I and Type II Errors**

### **Type I Error (False Positive)**

* Rejecting a true null hypothesis
* Probability = α
* Cause: random noise interpreted as effect

Example: “We conclude the drug works, but it actually doesn’t.”

---

### **Type II Error (False Negative)**

* Failing to reject a false null hypothesis
* Probability = β
* Related to:

  * Low sample size
  * High variability
  * Weak effect size
  * Bad study design

Example: “We conclude the drug does not work, but it actually does.”

---

### **Power of a Test**

Power = 1 − β
= probability of detecting true effect.

---

# **11. Multiple Testing Problem (Alpha Inflation)**

If you perform many hypothesis tests, the chance of getting *at least one* false positive increases dramatically.

### **Alpha Inflation**

If each test has α = 0.05:

* 1 test → 5% false positive
* 20 tests → ~64% chance of getting at least one false positive
* 100 tests → >99%

This happens in:

* Multiple predictors in a model
* Multiple sub-group analyses
* Lots of feature engineering
* Trying many ML models
* Exploratory data analysis

---

# **12. Fixing Alpha Inflation**

## **(a) Make α More Stringent**

E.g., instead of 0.05, use 0.01 or 0.001.

---

## **(b) Bonferroni Correction**

Very conservative.

If m tests are done:
**Corrected α = α/m**

Example:
Testing 10 hypotheses → α = 0.05/10 = 0.005

Reduces false positives, but increases Type II errors (false negatives).

---

## **(c) False Discovery Rate (FDR)**

Less strict than Bonferroni.
Controls expected proportion of false positives among rejected hypotheses.

**Benjamini–Hochberg Procedure**:

* Sort p-values
* Compare each p-value to a scaled threshold
* Maintains balance between false positives and power

Used in:

* Genomics
* ML model selection with many features
* Large-scale A/B testing

---

# **13. Summary Table**

| Concept                  | Meaning                                  | Key Issues                      |
| ------------------------ | ---------------------------------------- | ------------------------------- |
| **Null Hypothesis (H₀)** | No effect/difference                     | Default assumption              |
| **Alternative (H₁)**     | Effect/difference exists                 | Directional or not              |
| **p-value**              | Prob(data / H₀ true)                     | Often misinterpreted            |
| **α (alpha)**            | Allowed Type I error                     | 0.05 by tradition               |
| **Type I error**         | False positive                           | Controlled by α                 |
| **Type II error**        | False negative                           | Controlled by sample size/power |
| **Permutation Test**     | Shuffle labels to test difference        | Non-parametric                  |
| **Bootstrap Test**       | Resample with replacement                | Great for CIs + testing         |
| **Alpha Inflation**      | Multiple tests → more false positives    | Needs correction                |
| **Bonferroni**           | Divide α by number of tests              | Very conservative               |
| **FDR (BH)**             | Controls proportion of false discoveries | More power                      |

---

# **14. Final Intuition**

Hypothesis testing is about:

* Checking whether observed differences are explainable by **chance**
* Avoiding false conclusions due to randomness or over-analysis
* Understanding *statistical* vs *practical* significance
* Using corrections when doing many analyses

---

---

# **📘 Common Hypothesis Tests **

---

# **1. t-Tests**

t-tests are used when comparing means, especially when sample size is small or population variance is unknown.

## **1.1 Why t Distribution?**

* When population variance **σ² is unknown**, we estimate it from sample data.
* The statistic:
  $$
  t = \frac{\bar X - \mu_0}{s / \sqrt{n}}
  $$
  follows a **Student’s t-distribution**, which accounts for extra uncertainty.

### **Properties**

* Symmetric, bell-shaped, similar to normal.
* *Heavier tails*: accounts for small sample size.
* Approaches normal distribution as n → ∞.

---

## **1.2 Types of t-Tests**

### **(a) One-Sample t-Test**

Tests if the sample mean differs from a known population mean.

### **(b) Two-Sample t-Test**

Used to compare mean of two independent groups.
Assumes independence + similar variance (optional variant compensates).

Variants:

* **Independent t-test (equal variances)** → Student’s t
* **Welch’s t-test (unequal variances)** → Welch's approximation for DF

### **(c) Paired t-Test**

Used for **before/after** or matched subjects.
Works on difference scores.

---

## **1.3 Standardization**

The t-statistic is a standardized score:
$$
t = \frac{\text{Observed Difference}}{\text{Standard Error of Difference}}
$$

Higher |t| values → stronger evidence against H₀.

---

# **2. ANOVA (Analysis of Variance)**

ANOVA extends the t-test to **comparing means of more than 2 groups**.

Instead of comparing means pairwise, ANOVA compares **variances** attributable to group differences.

### **Intuition**

Under H₀: all groups have same population mean.
If between-group variance is much greater than within-group variance, H₀ likely false.

---

## **2.1 Decomposition of Variance in ANOVA**

Total variability (SST) is decomposed as:

$$
SST = SSB + SSE
$$

Where:

* **SSB (Between-group Sum of Squares)**
  Variation due to group differences → *group effect*.

* **SSE (Within-group Sum of Squares)**
  Variation within groups → *residual effect*.

---

## **2.2 F-Statistic**

$$
F = \frac{\text{Mean Square Between (MSB)}}{\text{Mean Square Error (MSE)}}
$$

Where:

* $$MSB = SSB / df(_{between})$$
* $$MSE = SSE / df(_{within})$$

### **Distribution**

The statistic follows an **F-distribution** under the null.

### **Degrees of Freedom**

* $$df(_{between} = k - 1)** (k = number of groups)$$
* $$df(_{within} = N - k)** (N = total samples)$$

---

## **2.3 One-Way ANOVA**

* One factor (e.g., different treatments)
* Compares means across k groups.

---

## **2.4 Two-Way ANOVA**

Includes **two factors**:

Example:

* Factor A: drug type
* Factor B: dosage level

Allows testing:

1. **Main effect of A**
2. **Main effect of B**
3. **Interaction effect** A×B

Interpretation:

* Interaction: effect of A changes depending on level of B.

---

## **2.5 Resampling Approach for ANOVA**

Permutation-based ANOVA:

1. Randomly shuffle group labels.
2. Recompute F-statistic.
3. Compare observed F to permutation distribution.

Advantages:

* Non-parametric
* No need for normality/variance homogeneity assumptions

---

# **3. Chi-Square Test**

Chi-square tests are used for **categorical data**.

## **3.1 Use Cases**

* Testing independence of two categorical variables
* Goodness-of-fit test for categorical distribution
* Testing proportions in contingency tables

---

## **3.2 Chi-Square Statistic**

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

Where:

* $$O_i$$ = observed counts
* $$E_i$$ = expected counts under H₀

Large chi-square values → observed distribution unlikely under null.

---

## **3.3 Expected Counts under Null**

Example: Independence test for a 2×2 table:

$$
E_{ij} = \frac{(\text{Row Total}) \times (\text{Column Total})}{\text{Grand Total}}
$$

---

## **3.4 Degrees of Freedom**

* Goodness of fit: df = (k − 1)
* Independence test: df = (r − 1)(c − 1)

---

## **3.5 Resampling Approach**

Permutation version:

1. Shuffle labels of one variable
2. Recalculate chi-square
3. Build distribution under null
4. Compare observed stat to permutation distribution

Used when assumptions are violated, small cell counts, etc.

---

# **4. Shortfalls of Classical Hypothesis Testing**

* Over-reliance on **p-values**
* p < 0.05 does not imply practical significance
* Sensitive to sample size
* Multiple testing inflates Type I errors
* Assumption-heavy (normality, equal variances)
* Encourages **p-hacking**
* Binary decision (reject/accept) hides nuance
* Doesn’t provide probabilities of hypotheses

Modern alternatives:

* Bayesian methods
* Confidence intervals
* Estimation/Effect-size focus

---

# **5. Multi-Armed Bandit Algorithms**

### Goal:

Balance **exploration** (learn which treatment is best) and **exploitation** (use best treatment so far).

Used in:

* Adaptive A/B testing
* Recommendation systems
* Online advertising
* Clinical trials

---

## **5.1 Epsilon-Greedy Algorithm**

A simple multi-armed bandit strategy.

### **Process:**

* With probability **ε** → explore (choose random arm)
* With probability **1 − ε** → exploit (choose best so far)

Typical choice:

* ε = 0.1 (10% exploration)
* ε decays over time (epsilon-decay)

Pros:

* Very simple
* Works reasonably well

Cons:

* Can waste time exploring suboptimal arms
* Does not consider uncertainty in estimates

---

## **5.2 Other Multi-Armed Bandit Approaches**

(Some for context)

* Upper Confidence Bound (UCB1)
* Thompson Sampling
  These are more efficient than epsilon-greedy.

---

# **6. Power, Sample Size, Alpha, and Effect Size**

These four quantities are interrelated:

| Quantity                   | Meaning                              |
| -------------------------- | ------------------------------------ |
| **Power (1 − β)**          | Probability of detecting true effect |
| **α (Significance level)** | Max allowed false positive rate      |
| **Effect size**            | Magnitude of true effect             |
| **Sample size (n)**        | Number of observations               |

---

# **6.1 Relationships Between the Four**

### **(1) Larger Effect Size → Higher Power**

A big effect is easier to detect.

---

### **(2) Larger Sample Size → Higher Power**

More data reduces standard error.

---

### **(3) Smaller α → Lower Power**

A stricter threshold makes detection harder.

---

### **(4) Higher Power → Requires Larger n**

Typical target: **80% power**.

---

# **6.2 Finding One Component When Three Are Known**

### **To find required sample size (n):**

Given:

* desired power
* effect size
* α

→ use power analysis formulas (e.g., Cohen's d for mean differences).

---

### **To find power:**

Given:

* n
* effect size
* α

→ compute probability of rejecting H₀ under true effect.

---

### **To find detectable effect size:**

Given:

* n
* power
* α

→ solve for minimum effect size detectable.

---

### **To find α needed for desired power:**

Given:

* n
* effect size
* desired power
  → tighter α → need larger n.

Common libraries:

* Python: `statsmodels.stats.power`
* R: `pwr` package

---

# **6.3 Visual Intuition**

* Smaller effect = distributions overlap → harder to detect → need large n
* Larger effect = easy to detect even with small n

A good experimental design requires balancing cost (sample size) vs statistical goals.

---

# **7. Summary Table**

| Test                 | Used For           | Statistic | Distribution           | Notes               |
| -------------------- | ------------------ | --------- | ---------------------- | ------------------- |
| **t-test**           | Mean differences   | t         | Student’s t            | For 1 or 2 groups   |
| **ANOVA**            | >2 means           | F         | F-distribution         | Decomposes variance |
| **Chi-square**       | Categorical counts | χ²        | Chi-square             | Independence or GOF |
| **Permutation Test** | Non-parametric     | Any       | Generated empirically  | Fewer assumptions   |
| **Bootstrap Test**   | Non-parametric     | Any       | Bootstrap distribution | CI + inference      |

---