#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "raw"

print("Project Root:", PROJECT_ROOT)
print("Data Directory:", DATA_DIR)


# ## Step 1 – Load Both Datasets
# 
# Load the English and Japanese CSV files.
# 
# The English dataset has already been explored in Notebook 01.
# 
# In this notebook, the English dataset is used only as the reference for comparison.

# In[4]:


english = pd.read_csv(DATA_DIR / "EN_Card_Data.csv")
japanese = pd.read_csv(DATA_DIR / "JP_Card_Data.csv")

print("Datasets loaded successfully.")


# ## Step 2 – Compare Dataset Dimensions
# 
# Verify that both datasets contain the same number of rows and columns.

# In[5]:


print("English Dataset:", english.shape)
print("Japanese Dataset:", japanese.shape)


# ## Step 3 – Compare Column Names
# 
# Before analyzing the Japanese dataset, we must verify that both datasets share the exact same schema.
# 
# If the column names match, the same preprocessing and feature engineering pipeline can be used for both datasets.

# In[6]:


english.columns


# In[7]:


japanese.columns


# In[8]:


english.columns.equals(japanese.columns)


# ## Observation 2
# 
# Although the English and Japanese datasets contain the same number of columns, the column names are translated into their respective languages.
# 
# This is expected because the Japanese dataset is intended for Japanese players.
# 
# The identical dataset dimensions suggest that both files describe the same information using different languages.

# ## Step 4 – Compare Data Types
# 
# Even though the column names are translated, we expect the data types to remain consistent.
# 
# Matching data types indicate that both datasets can be processed using the same feature engineering pipeline.

# In[10]:


english.dtypes


# In[11]:


japanese.dtypes


# In[12]:


print("English dtypes")
print(english.dtypes)

print("\nJapanese dtypes")
print(japanese.dtypes)


# ## Observation 3
# 
# Both datasets share the same structural layout and data types.
# 
# The only major difference is that the Japanese dataset contains translated field names and translated card content.
# 
# This confirms that a language-independent preprocessing pipeline can be developed for the AI Battle Agent.

# ## Step 5 – Compare Basic Statistics
# 
# Before comparing individual cards, we first compare the overall statistics of both datasets.
# 
# This helps verify that both files contain the same amount of information and confirms that they represent the same collection of Pokémon cards in different languages.

# In[13]:


print("English Dataset")
english.describe(include="all")


# In[14]:


print("Japanese Dataset")
japanese.describe(include="all")


# ## Observation 4
# 
# The English and Japanese datasets share identical dimensions and numerical statistics.
# 
# However, several categorical fields differ because they are localized for different Pokémon TCG releases.
# 
# For example:
# 
# - Expansion names differ.
# - Collection numbering differs.
# - Card text is translated.
# - Category names are localized.
# 
# These differences are expected and do not affect the overall dataset structure.
# 
# The numerical attributes (such as HP and Retreat Cost) remain consistent, making it possible to build a language-independent AI preprocessing pipeline.

# ## Step 6 – Compare Missing Values
# 
# Many Pokémon TCG fields are optional.
# 
# For example:
# 
# - Energy cards do not have HP.
# - Trainer cards do not have attacks.
# - Stadium cards do not have Retreat Cost.
# 
# The missing-value pattern should therefore be nearly identical between the English and Japanese datasets.

# In[15]:


english.isnull().sum()


# In[16]:


japanese.isnull().sum()


# In[17]:


comparison = pd.DataFrame({
    "English Missing": english.isnull().sum(),
    "Japanese Missing": japanese.isnull().sum()
})

comparison


# In[20]:


comparison = pd.DataFrame({
    "English Column": english.columns,
    "Japanese Column": japanese.columns,
    "English Missing": english.isnull().sum().values,
    "Japanese Missing": japanese.isnull().sum().values
})

comparison


# In[21]:


comparison["Difference"] = (
    comparison["English Missing"]
    - comparison["Japanese Missing"]
)

comparison


# ## Observation 5
# 
# The English and Japanese datasets exhibit very similar missing-value patterns.
# 
# Most structural fields have identical or nearly identical counts of missing values.
# 
# Small differences (such as Expansion) are expected because the English and Japanese Pokémon TCG releases contain localized metadata.
# 
# Overall, both datasets preserve the same underlying card structure, allowing a common preprocessing pipeline to support multilingual analysis.

# # Step 7 – Compare Equivalent Cards
# 
# ## Objective
# 
# The previous analyses verified that both datasets have similar structures.
# 
# In this section, we compare sample cards from both datasets to observe how Pokémon card information is represented in English and Japanese.
# 
# Rather than focusing on translation quality, our goal is to verify that both datasets describe the same game mechanics using different languages.

# In[22]:


english.head(3)


# In[23]:


japanese.head(3)


# ## Observation 6
# 
# Although the text is written in different languages, both datasets follow the same logical structure.
# 
# Each record contains:
# 
# - Card identifier
# - Card name
# - Expansion
# - Card stage
# - HP
# - Pokémon type
# - Weakness
# - Resistance
# - Retreat Cost
# - Move information
# 
# This confirms that both datasets represent equivalent Pokémon Trading Card Game data.

# # Notebook Summary
# 
# ## Objective
# 
# Compare the English and Japanese Pokémon TCG datasets.
# 
# ## Findings
# 
# ✔ Both datasets contain 2,022 records.
# 
# ✔ Both datasets contain 17 primary features.
# 
# ✔ Both datasets share the same logical structure.
# 
# ✔ Numerical attributes are consistent.
# 
# ✔ Text fields are localized into different languages.
# 
# ✔ Missing-value patterns are nearly identical.
# 
# ## Conclusion
# 
# The English and Japanese datasets describe the same Pokémon Trading Card Game collection while supporting different languages.
# 
# Therefore, a single preprocessing pipeline can be developed for multilingual AI battle analysis.

# In[ ]:




