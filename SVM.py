# %% [markdown]
# Support Vector Machine Implementaion

# %%
# Read data_Set
import pandas as pd


# %%
data = pd.read_csv('hepatitis.csv', na_values='?')
data.head()

# %%
data.shape

# %%
data.info()

# %%
data.isna().sum()

# %%
print(type(data.nunique()[0]))
# %%
data.nunique()[0]

# %%
column_names = data.columns
column_names

# %%
data.drop(['ID'], axis=1, inplace=True)
cat_cols = data.columns[data.nunique() < 5]

# %%
num_cols = data.columns[data.nunique() >= 5]

# %%
col_names = list(data.columns)
col_names.remove('target')
X, y = data[col_names], data['target']
X.head
# %%
X.shape

# %%
from sklearn.preprocessing import Imputer


# %%
