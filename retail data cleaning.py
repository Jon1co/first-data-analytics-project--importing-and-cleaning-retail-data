import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = 'plotly_dark'

df=pd.read_csv("new_retail_data copy.csv")
print (df.shape) #shows the number of rows and colums
print (df.head()) #shows the top 5 rows
print (df.tail()) #shows the bottom 5 rows
print (df.info()) #shows the columns and data and number of values that arent missing Non-Null
print (df.describe()) #shows the summary statistics of the data
print (df.isnull().sum()) #shows the number of missing values in each column
df.duplicated().sum()
df[df.duplicated].head()

#Rename The city Columns That Having in Accurate Values

# Replace incorrect city names

df['City'] = df['City'].str.replace("MÃ¼nster", "Münster")
df['City'] = df['City'].str.replace("St. John's", "St. Johns")
df['City'] = df['City'].str.replace("DÃ¼sseldorf", "Düsseldorf")
df['City'] = df['City'].str.replace("KÃ¶ln", "Kö")

df['Zipcode'].info()
mean_zipcode = df['Zipcode'].mean()
mean_zipcode
df.loc[:, 'Zipcode'] = df['Zipcode'].fillna(mean_zipcode)

# Convert 'Zipcode' to integers
df['Zipcode'] = df['Zipcode'].astype(np.int64)

# Count occurrences of zip codes with 4 digits
zip_4_digits_count = (df['Zipcode'].astype(str).str.len() == 4).sum()

# Count occurrences of zip codes with 5 digits
zip_5_digits_count = (df['Zipcode'].astype(str).str.len() ==5).sum()

total_zipcodes = len(df)

zip_4_digits_percentage = (zip_4_digits_count / total_zipcodes) * 100

zip_5_digits_percentage = (zip_5_digits_count / total_zipcodes) * 100
print("Number of 4-digit zip codes:", zip_4_digits_count)
print("Number of 5-digit zip codes:", zip_5_digits_count)
print("Percentage of 4-digit zip codes:", zip_4_digits_percentage)
print("Percentage of 5-digit zip codes:", zip_5_digits_percentage)
#here we check first  persentage wich one zipcode has more in data set based on it we remove less persentage number zip code bcz it is not consistance...

df['Zipcode'] = df['Zipcode'].astype(str)

import random
# Trim any leading or trailing whitespace
df['Zipcode'] = df['Zipcode'].str.strip()

# Count the number of 5-digit zipcodes
zip_5_digits = df['Zipcode'].str.len().eq(5).sum()

# Identify 4-digit zipcodes
zip_4_digits = df['Zipcode'].str.len().eq(4)

# If there are any 4-digit zipcodes, append a random digit
if zip_4_digits.any():
    df.loc[zip_4_digits, 'Zipcode'] = df.loc[zip_4_digits, 'Zipcode'].apply(lambda x: x + str(random.randint(0, 9)))
    mean_zipcode = df['Zipcode'].astype(float).mean()
df['Zipcode'] = df['Zipcode'].fillna(mean_zipcode)

# Convert 'Zipcode' to integers
df['Zipcode'] = df['Zipcode'].astype(int)
print (df.head())
df['Zipcode'].info()
# Rename country names
country = {"USA": "United States of America", "UK": "United Kingdom"}
df['Country'] = df['Country'].map(country).fillna(df['Country'])  #we also fill nan values with country name ...using fillna
print (df.head())
# Calculate the mean of 'Transaction_ID' and 'Customer_ID'
mean_transaction_id = df['Transaction_ID'].astype(float).mean()
mean_customer_id = df['Customer_ID'].astype(float).mean()

# Fill NaN values with the mean
df.loc[:, 'Transaction_ID'] = df['Transaction_ID'].fillna(mean_transaction_id)
df.loc[:, 'Customer_ID'] = df['Customer_ID'].fillna(mean_customer_id)

# Convert 'Transaction_ID' and 'Customer_ID' to integers
df['Transaction_ID'] = df['Transaction_ID'].astype(int)
df['Customer_ID'] = df['Customer_ID'].astype(int)

# Fill NaN values in 'Phone' with the mean
df.loc[:, 'Phone'] = df['Phone'].fillna(df['Phone'].astype(float).mean())

# Convert 'Phone' to integers
df['Phone'] = df['Phone'].astype(int)
print(df.head())

#impute missing values for age, total purchases, and year to with means as floats
df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].astype(float).mean())
df.loc[:, 'Total_Purchases'] = df['Total_Purchases'].fillna(df['Total_Purchases'].astype(float).mean())
df.loc[:, 'Year'] = df['Year'].fillna(df['Year'].astype(float).mean())

df.loc[:, 'Transaction_ID'] = df['Transaction_ID'].fillna(mean_transaction_id)
df.loc[:, 'Customer_ID'] = df['Customer_ID'].fillna(mean_customer_id)
df.loc[:, 'Phone'] = df['Phone'].fillna(df['Phone'].astype(float).mean())

# Remove rows with negative Total_Purchases, Amount, and Total_Amount values
df = df[(df['Total_Purchases'] >= 0) & (df['Amount'] >= 0) & (df['Total_Amount'] >= 0)]

#converting date , year ,time into datetime datatype formate

df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
df['Time'].fillna(df['Time'])

#useing interpolate for filling missing values with available values
df['Total_Purchases'] = df['Total_Purchases'].interpolate(direction='both')
df['Amount'] = df['Amount'].interpolate(direction='both')

df['Total_Amount'] = df['Total_Purchases'] * df['Amount']

# Calculate the percentage of null values in each column
null_percentage = (df.isnull().sum() / len(df)) * 100
print ("the null percentages are")
print (null_percentage)

# Fill null values in numerical columns with mean                                      #mean fill values like :if you have 1,2,NAN,4,5 then avg 1to5(1+2+4+5)/4=values...is fill in our NAN val...
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
# Fill null values in categorical columns with mode                                      #same in Categorical col of you have more time some values then it was getting mode from it ...
cat_columns = df.select_dtypes(include=['object']).columns
for col in cat_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
print(df.isnull().sum())
print(df.shape)
print(df.info)
print (df[df['Total_Purchases']==2].head(6))
print(df.duplicated().sum())
print(df.columns)
print (df[df.duplicated()].head())
# Before removing duplicates
print("Shape of DataFrame before removing duplicates:", df.shape)

# Keep only the first occurrence of each duplicated row and remove the rest so it was perfectly removed duplicates from df...
df.drop_duplicates(subset=['Transaction_ID', 'Customer_ID', 'Name', 'Email', 'Phone', 'Address', 'City', 'State', 'Zipcode', 'Country', 'Age', 'Gender', 'Income', 'Customer_Segment', 'Date', 'Year', 'Month', 'Time', 'Total_Purchases', 'Amount', 'Total_Amount', 'Product_Category', 'Product_Brand', 'Product_Type', 'Feedback', 'Shipping_Method', 'Payment_Method', 'Order_Status'], keep='first', inplace=True)

# After removing duplicates
print("Shape of DataFrame after removing duplicates:", df.shape)
print(df.duplicated().sum())
#save the newly cleaned data into a csv file
df.to_csv("cleaned_retail_data.csv", index=False)