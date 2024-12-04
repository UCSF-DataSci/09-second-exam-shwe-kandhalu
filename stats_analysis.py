# Perform statistical analysis on both outcomes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf 
from statsmodels.formula.api import ols, mixedlm

# 1. Analyze walking speed:
# - Multiple regression with education and age
# - Account for repeated measures
# - Test for significant trends

df = pd.read_csv('ms_data_new.csv')

# multiple regression with education and age
model = sm.OLS.from_formula('walking_speed ~ age + C(education_level)', df).fit()
print("Multiple regression results for walking speed:")
# summary of regression
print(model.summary())

# repeated measures
mixed_model = mixedlm('walking_speed ~ age + C(education_level)', df, groups=df['patient_id'])
mixed_model_results = mixed_model.fit()
print("\nMixed-effects model results for walking speed:")
print(mixed_model_results.summary())

# test for significant trends
df['education_level'] = df['education_level'].astype('category')
anova_results = sm.stats.anova_lm(model, typ=2)
print("\nANOVA results for walking speed based on age and education:")
print(anova_results)

# The fixed-effects coefficients (Coef.) describe the influence of each predictor on walking_speed.
# The intercept of 5.6232 is the predicted walking speed for the reference category. 
# Education Level:
# Graduate: Increases walking speed by 0.3894 units
# High School: Decreases walking speed by 0.8080 units (significant).
# Some College: Decreases walking speed by 0.4036 units (significant).
# For each additional year of age, walking speed decreases by 0.0303 units (significant).

# 2. Analyze costs:
# - Simple analysis of insurance type effect
# - Box plots and basic statistics
# - Calculate effect sizes

# mean, median, and standard deviation of costs
cost_summary = df.groupby('insurance_type')['visit_cost'].agg(['mean', 'median', 'std'])
print(cost_summary)

# boxplot of visit costs
sns.boxplot(x='insurance_type', y='visit_cost', data=df)
plt.title("Visit Costs by Insurance Type")
plt.xlabel("Insurance Type")
plt.ylabel("Visit Cost")
plt.savefig('boxplot_visit_cost')

# histogram of visit costs across insurance types
plt.hist('visit_cost', df, bins=25)
plt.savefig('visit_cost_hist.png')

# effect size calculation
# Cohen's d for Basic vs Premium
group1 = df[df['insurance_type'] == 'Basic']['visit_cost']
group2 = df[df['insurance_type'] == 'Premium']['visit_cost']

# Cohen's d calculation
mean1, mean2 = group1.mean(), group2.mean()
std1, std2 = group1.std(), group2.std()
pooled_std = np.sqrt(((std1**2 + std2**2) / 2))
cohens_d = (mean1 - mean2) / pooled_std

print("\nCohen's d for insurance type effect on costs:")
print(cohens_d)

# the effect size is very large and is negative
# the difference in each price point for each insurance is very varied, so this makes sense

# 3. Advanced analysis:
# - Education age interaction effects on walking speed
# - Control for relevant confounders
# - Report key statistics and p-values

# education age interaction effects on walking speed
model_interaction = smf.mixedlm('walking_speed ~ age * C(education_level)', df, groups=df['patient_id'])
interaction_results = model_interaction.fit()
print(interaction_results.summary())

# The interaction between age and high school education is significant. 
# This suggests that age affects walking speed differently for people with 
# a high school education compared to others.

# control for relevant confounders
confounders = sm.add_constant(df[['age', 'education_level_Some_College', 'education_level_Graduate']])

# Fit the model
model_confounders = sm.OLS(y, confounders).fit()

# Print the summary
print("\nRegression results with age as a confounder:")
print(model_confounders.summary())

# the intercept coefficient is 5.657 This is the estimated walking speed when age is zero
# the model predicts a walking speed of 5.657 for this group. the p-value is 0.000, indicating 
# that the intercept is statistically significant.

# Education Level:
# Graduate: Faster walking speed (0.365 units faster).
# High School: Slower walking speed (0.874 units slower).
# Some College: Slower walking speed (0.433 units slower).

# Age:
# Older individuals tend to have slower walking speeds, 
# with a significant decrease of 0.031 units per year.