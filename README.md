# Stages-of-Data-Science
## Aim
To Analyze a data set with Various stages of data science.

## Steps
1.Choose your own dataset

2.Perform Data Preprocessing

3.Implement Data analysis.

4.Perform Feature Engineering process

5.Implement any two Advanced data Visualization 

## Code
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

tips = sns.load_dataset('tips')

print("Dataset preview:\n", tips.head())
print("\nDataset info:\n")
tips.info()

print("\nMissing values:\n", tips.isnull().sum())

le = LabelEncoder()
tips['sex'] = le.fit_transform(tips['sex'])
tips['smoker'] = le.fit_transform(tips['smoker'])
tips['day'] = le.fit_transform(tips['day'])
tips['time'] = le.fit_transform(tips['time'])

scaler = StandardScaler()
tips[['total_bill', 'tip', 'size']] = scaler.fit_transform(tips[['total_bill', 'tip', 'size']])

print("\nDescriptive statistics:\n", tips.describe())
print("\nCorrelation matrix:\n", tips.corr())
sns.heatmap(tips.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips['group_type'] = tips['size'].apply(lambda x: 'small' if x <= 2 else 'large')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip_pct', data=tips, hue='day')
sns.regplot(x='total_bill', y='tip_pct', data=tips, scatter=False, color='red')
plt.title("Tip Percentage vs Total Bill with Regression Line")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='tip_pct', hue='group_type', data=tips)
plt.title("Tip Percentage Distribution by Day and Group Type")
plt.show()
```
## Output
![Screenshot 2024-11-15 114423](https://github.com/user-attachments/assets/2b7ac997-12c2-4f52-a323-b79e14212a58)

![Screenshot 2024-11-15 114437](https://github.com/user-attachments/assets/cc8f5e1c-c023-41ea-948d-8cd219760a22)
![Screenshot 2024-11-15 114454](https://github.com/user-attachments/assets/329c311a-5d82-4434-90ce-72dce75f4f2a)


## Result
To Analyze a data set with Various stages of data science is done successfully.
