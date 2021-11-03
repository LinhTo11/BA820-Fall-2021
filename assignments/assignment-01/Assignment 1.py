import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


forums = pd.read_pickle("C:/Users/LinhTo/Desktop/BU/Class - Teacher/BA 820/BA820-Fall-2021/assignments/assignment-01/forums.pkl")
pd.set_option('display.max_columns', False)
forums.head()

# Data cleaning
forums.shape
forums.dtypes
forums.isna().sum()

## % of rows missing in each column
for column in forums.columns:
    percentage = forums[column].isnull().mean()
    print(f'{column}: {round(percentage*100, 2)}%')

## Visualize using missingno package
import missingno as msno
msno.matrix(forums)
plt.show(forums)

# Statistics 
numerics = forums.loc[: ,"e0":"e295"].describe()
numerics.columns
sns.regplot(x='e0',data=numerics)
plt.show()