# Using the cleaned data and insurance category file from Question 1:

import pandas as pd
import numpy as np 
import random

# 1. Load and structure the data:
# - Read the processed CSV file
# - Convert visit_date to datetime
# - Sort by patient_id and visit_date

# read in processed CSV file
df = pd.read_csv("ms_data.csv")
df = df.dropna() # drop missing values
df = df.drop_duplicates() #drop duplicates

# convert visit_data to datetime
df['visit_date'] = pd.to_datetime(df['visit_date'])
print(df.dtypes)

# sort by patient_id and visit_date
df = df.sort_values(by = ['patient_id', 'visit_date'])

# 2. Add insurance information:
# - Read insurance types from insurance.lst
# - Randomly assign (but keep consistent per patient_id)
# - Generate visit costs based on insurance type:
# - Different plans have different effects on cost
# - Add random variation

insurance_types = pd.read_csv('insurance.lst', header=None, names=['insurance_type'])
print(insurance_types)

# create dictionary to store consistent insurance assignment
unique_patients = df['patient_id'].unique()
insurance_mapping = {patient: np.random.choice(insurance_types['insurance_type']) for patient in unique_patients}

# map insurance type to patient IDs
df['insurance_type'] = df['patient_id'].map(insurance_mapping)

# cost per insurance type
costs = {'Basic': 100, 'Premium': 200, 'Platinum': 300}

# random variation
np.random.seed(189)
df['visit_cost'] = df['insurance_type'].map(costs) + np.random.normal(0, 10, size=len(df))

# 3. Calculate summary statistics:
# - Mean walking speed by education level
# - Mean costs by insurance type
# - Age effects on walking speed

# Mean walking speed by education level
meanWalkSpeed = df.groupby('education_level')['walking_speed'].mean()
print("Mean walking speed by education level:", meanWalkSpeed)

# Mean costs by insurance type
meanCostIns = df.groupby('insurance_type')['visit_cost'].mean()
print("Mean costs by insurance type:", meanCostIns)

# Age effects on walking speed
# separate into bins for better analysis
bins = [0, 30, 50, 70, 100]
labels = ['0-30', '31-50', '51-70', '71-100']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

ageOnWalkSpeed = df.groupby('age_group')['walking_speed'].mean().reset_index()
print("Age effects on walking speed:")
print(ageOnWalkSpeed)

# extra: seeing relationship between walking speed and the month of visit date
df['month'] = df['visit_date'].dt.month

# group by month and calculate mean walking speed
monthly_speed = df.groupby('month')['walking_speed'].mean().reset_index()
print(monthly_speed)