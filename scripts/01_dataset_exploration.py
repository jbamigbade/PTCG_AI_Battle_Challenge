#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 200)

PROJECT_ROOT = Path.cwd().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

print("Project Root:", PROJECT_ROOT)
print("Data Directory:", DATA_DIR)


# In[2]:


cards = pd.read_csv(DATA_DIR / "EN_Card_Data.csv")


# In[3]:


cards.shape


# # Observation 1
# 
# The official English Pokémon TCG dataset contains:
# 
# - **2,022 cards**
# - **17 attributes (columns)**
# 
# This dataset will serve as the primary knowledge base for our AI Training Agent.

# In[6]:


cards.columns


# ## Observation 2
# 
# The dataset contains 17 fields describing each Pokémon card.
# 
# Next, we will examine each field to determine how it contributes to AI decision-making, deck construction, and battle strategy.

# In[8]:


cards.head()


# ## Observation 3
# 
# The first five records in the dataset are **Basic Energy** cards rather than Pokémon.
# 
# This indicates that the dataset stores **Pokémon, Trainer, and Energy cards together in one table**.
# 
# Fields that do not apply to a particular card type (such as HP or attacks for Energy cards) are represented as **NaN (Not a Number)** values.

# In[9]:


cards["Category"].value_counts()


# ## Observation 4
# 
# The `Category` column does **not** represent the broad card type (Pokémon, Trainer, or Energy).
# 
# Instead, it describes special classifications and mechanics such as:
# 
# - Trainer's Pokémon
# - Ancient
# - Future
# - Tera Pokémon
# - Fossil
# - Technical Machine
# 
# This means we must use other columns to distinguish between Pokémon, Trainer, and Energy cards.

# In[10]:


cards["Stage (Pokémon)/Type (Energy and Trainer)"].value_counts()


# ## Observation 5
# 
# The **Stage (Pokémon) / Type (Energy and Trainer)** column is one of the most important fields in the dataset.
# 
# It identifies the functional role of each card, including:
# 
# ### Pokémon
# - Basic Pokémon
# - Stage 1 Pokémon
# - Stage 2 Pokémon
# 
# ### Trainer Cards
# - Item
# - Supporter
# - Stadium
# - Pokémon Tool
# 
# ### Energy Cards
# - Basic Energy
# - Special Energy
# 
# This field will allow the AI to determine the legal actions available for each card during gameplay.

# In[11]:


cards.isnull().sum()


# ## Observation 6
# 
# The dataset contains several columns with missing values.
# 
# These missing values are **expected** and are primarily caused by differences between Pokémon, Trainer, and Energy cards.
# 
# Examples include:
# 
# - Energy cards do not have HP or attacks.
# - Trainer cards do not have retreat costs or weaknesses.
# - Basic Pokémon do not have a previous evolution stage.
# 
# Therefore, these missing values represent **game mechanics**, not data quality issues.

# In[12]:


cards.info()


# ## Observation 7
# 
# The dataset contains **2,022 records** and **17 columns**.
# 
# ### Data Types
# 
# - **13 text (`str`) columns**
# - **2 integer (`int64`) columns**
# - **2 numeric (`float64`) columns**
# 
# The numeric columns (`HP` and `Retreat`) are stored as floating-point values because they contain missing values (`NaN`).
# 
# The entire dataset occupies only **268.7 KB** of memory, making it lightweight and efficient for AI processing.

# In[13]:


cards[cards["Stage (Pokémon)/Type (Energy and Trainer)"] == "Basic Pokémon"].head()


# ## Observation 8
# 
# The dataset is organized with **one row per attack**, not one row per Pokémon card.
# 
# Cards that have multiple attacks appear multiple times.
# 
# For example:
# 
# - Team Rocket's Kangaskhan ex appears twice because it has two attacks.
# - Pinsir appears twice because it has two attacks.
# 
# This design makes it easier for an AI agent to evaluate each available attack independently during battle.

# In[14]:


cards[cards["Card Name"] == "Pinsir"]


# ## Observation 9
# 
# Cards are uniquely identified by **Card ID**.
# 
# Multiple rows may share the same Card ID because each row represents a different attack.
# 
# For example:
# 
# Card ID 25 (Pinsir)
# 
# - Slow Crunch
# - Superpowered Horns
# 
# During battle, the AI should retrieve all rows with the same Card ID to determine every legal attack available for that Pokémon.

# In[15]:


cards["Card ID"].nunique()


# ## Observation 10
# 
# Although the dataset contains **2,022 rows**, there are only **1,267 unique Card IDs**.
# 
# This confirms that multiple rows may belong to the same card.
# 
# Each row represents one attack associated with that card.
# 
# Therefore:
# 
# - **Card ID** uniquely identifies a card.
# - Multiple rows with the same Card ID represent different attacks.
# 
# The AI should organize information around **Card IDs**, not individual rows.

# In[16]:


rows = len(cards)
unique_cards = cards["Card ID"].nunique()

print(f"Total rows: {rows}")
print(f"Unique cards: {unique_cards}")
print(f"Duplicate attack rows: {rows - unique_cards}")
print(f"Average rows per card: {rows / unique_cards:.2f}")


# ## Observation 11
# 
# The dataset contains:
# 
# - **2,022 total rows**
# - **1,267 unique cards**
# - **755 additional attack rows**
# 
# On average, each card has **1.60 attack records**.
# 
# This confirms that the dataset stores attacks separately rather than storing all attacks within a single row.
# 
# When building the AI, attack information should be grouped by **Card ID**.

# In[ ]:




