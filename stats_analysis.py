# Perform statistical analysis on both outcomes

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
import numpy as np
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Analyze walking speed:
# - Multiple regression with education and age
# - Account for repeated measures
# - Test for significant trends

df = pd.read_csv('ms_data_new.csv')

# make a dictionary for education values
edu_code = {'High School': 1, 'Some College': 2, 'Bachelors': 3, 'Graduate': 4}

# create a new column with just the education codes
df['education_level_code'] = df['education_level'].map(edu_code)

# multiple regression with education and age
model = smf.ols('walking_speed ~ age + education_level', df).fit()

# summary of regression
print(model.summary())

# random intercept for patient_id
mixed_model = mixedlm('walking_speed ~ age + education_level', df, groups=df['patient_id'])
mixed_model_results = mixed_model.fit()

print(mixed_model_results.summary())

# The fixed-effects coefficients (Coef.) describe the influence of each predictor on walking_speed.
# The intercept of 5.621 is the predicted walking speed for the reference category. 
# Education Level:
# Graduate: Increases walking speed by 0.388 units
# High School: Decreases walking speed by 0.799 units (significant).
# Some College: Decreases walking speed by 0.403 units (significant).
# For each additional year of age, walking speed decreases by 0.030 units (significant).

# 2. Analyze costs:
# - Simple analysis of insurance type effect
# - Box plots and basic statistics
# - Calculate effect sizes

# mean, median, and standard deviation of costs
cost_summary = df.groupby('insurance_type')['visit_cost'].agg(['mean', 'median', 'std'])
print(cost_summary)

# boxplot of visit costs
sns.boxplot(x = 'insurance_type', y = 'visit_cost', df)
plt.savefig('boxplot.png')

# histogram of visit costs across insurance types
plt.hist('visit_cost', df, bins=25)
plt.savefig('visit_cost_hist.png')

# effect size calculation
def cohens_d(group1, group2):
    diff = abs(group1.mean() - group2.mean())
    pooled_sd = ((group1.std() ** 2 + group2.std() ** 2) / 2) ** 0.5
    return diff / pooled_sd

print(cohens_d(groups[0], groups[1]))  # Basic vs Premium

# the effect size is very large
# the difference in each price point for each insurance is very varied, so this makes sense

# 3. Advanced analysis:
# - Education age interaction effects on walking speed
# - Control for relevant confounders
# - Report key statistics and p-values