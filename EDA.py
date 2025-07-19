import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
 
# Load the dataset
df = pd.read_csv('train.csv')
# Display the first few rows of the dataset
print(df.head())
print(df.tail())
# Display the shape of the dataset
print(df.shape)
print(df.info())
# Display the summary statistics of the dataset 
print(df.describe())
#data cleaning
# Check for missing values  
df.isnull().sum()
# Drop rows with missing values 
df.dropna(inplace=True)
#check for duplicates
df.duplicated().sum()
# Drop duplicate rows
df.drop_duplicates(inplace=True)
#surviving passenger count
plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df, palette='Set1')
plt.title('Count of Survived vs Not Survived')  
plt.xticks([0, 1], ['Didn\'t Survive', 'Survived'])
plt.xlabel('Survival Status')
plt.ylabel('Count')
#survival on gender basis
survived_df = df[df['Survived'] == 1]
plt.subplot(2, 2, 2)
sns.countplot(x='Sex', data=survived_df, palette='Set2')
plt.title('male/female Survived Count')
plt.xlabel('survival Status')
plt.ylabel('Count')
#survival on age basis
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase vertical and horizontal spacing

plt.subplot(2, 2, 3)
sns.histplot(x='Age', data=survived_df, palette='viridis')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
#survival on class basis
plt.subplot(2, 2, 4)
ax = plt.gca()  # Get current axes
class_counts = survived_df['Pclass'].value_counts().sort_index()
colors = sns.color_palette('Set1', n_colors=3)
labels = ['Class 1', 'Class 2', 'Class 3']
patches, texts, autotexts = plt.pie(class_counts, labels=labels, startangle=140, colors=colors, autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.title('Survived Passengers by Class (Pie Chart)')
plt.legend(patches, labels, loc='best', title='Passenger Class', frameon=True)
ax.patch.set_edgecolor('black')  # Add black border to the subplot
ax.patch.set_linewidth(2)        # Set border thickness
plt.tight_layout()
plt.show()

#Age Distribution of Passengers
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
age_bins = list(range(0, 101, 10))
age_labels = [f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(len(age_bins)-1)]
survived_df['AgeGroup'] = pd.cut(survived_df['Age'], bins=age_bins, labels=age_labels, right=False)
age_group_counts = survived_df['AgeGroup'].value_counts().sort_index()
sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='viridis')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')


#gender distribution of passengers
plt.subplot(1, 2, 2)
gender_counts= df['Sex'].value_counts().sort_index()
labels = ['Male','female']
plt.pie(gender_counts, startangle=140, autopct='%1.1f%%', colors=sns.color_palette('bright6', n_colors=2))
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.title('Sex Distribution of Passengers')
plt.legend(labels, loc='best', title='Genders', frameon=True)
plt.tight_layout()
plt.show()

#fare distribution of passengers by class
plt.figure(figsize=(15, 6))

sns.barplot(x='Pclass', y='Fare', data=df, palette='viridis')
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')  
plt.show()

# Convert all non-numeric columns to numeric

df_encoded = df.copy()

# Encode 'Sex'
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})

# Encode 'Embarked'
if 'Embarked' in df_encoded.columns:
    df_encoded['Embarked'] = df_encoded['Embarked'].astype('category').cat.codes

# Encode 'Cabin' (use category codes, missing as -1)
if 'Cabin' in df_encoded.columns:
    df_encoded['Cabin'] = df_encoded['Cabin'].astype('category').cat.codes

# Encode 'Ticket' (use category codes)
if 'Ticket' in df_encoded.columns:
    df_encoded['Ticket'] = df_encoded['Ticket'].astype('category').cat.codes
# Encode 'Name' by its length
df_encoded['Name'] = df_encoded['Name'].apply(len)
# Now compute correlation matrix
corr_matrix = df_encoded.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Map for All Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# Save the cleaned and encoded DataFrame to a new CSV file
df_encoded.to_csv('train_encoded.csv', index=False)
print("Encoded DataFrame saved to 'train_encoded.csv'")
# Display the first few rows of the encoded DataFrame
print(df_encoded.head())
# Display the shape of the encoded DataFrame
print(df_encoded.shape) 