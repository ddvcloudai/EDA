import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub

# Download latest version
path = kagglehub.dataset_download("CooperUnion/cardataset")

# adding this to control timeout session
import requests

# Example with a longer timeout setting, this is optional
response = requests.get("https://path_to_your_resource", timeout=60)

print("Path to dataset files:", path)
df=pd.read_csv(f"/Users/divyadeepverma/.cache/kagglehub/datasets/CooperUnion/cardataset/versions/1/data.csv")
print(df.head(5))
print (df.dtypes)

# dropping the elements which are not required:
df=df.drop(['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors', 'Vehicle Size'], axis=1)
print(df.head(5))

#renaming of columns
df=df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode","highway MPG": "MPG-H", "city mpg": "MPG-C", "MSRP": "Price" })
print(df.head(5))

#deleting duplicate data
print(df.shape)
duplicate_rows_df=df[df.duplicated()]
print("no of duplicate rows: ", duplicate_rows_df)
print(df.count())
df=df.drop_duplicates()
print(df.head(5))
print(df.count())

#deleting null values

print(df.isnull().sum())
df=df.dropna() #dropping the null values
print(df.count())
print(df.isnull().sum()) #again checking for null values

# detecting and deleting outliers
sns.boxplot(x=df['Price'])
sns.boxplot(x=df['HP'])
sns.boxplot(x=df['Cylinders'])

q1=df.quantile(0.25)
q2=df.quantile(0.75)
iqr=q3-q1
print(iqr)

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df.shape)

# histogram plot for variable relationship

df.Make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by make")
plt.ylabel('Number of cars')
plt.xlabel('Make');


#heat map
plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
print(c)

#scatter plot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['HP'], df['Price'])
ax.set_xlabel('HP')
ax.set_ylabel('Price')
plt.show()
