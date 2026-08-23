#!/usr/bin/env python
# coding: utf-8

# # Notebook 03 – Data Preprocessing & Feature Engineering
# 
# ## Objective
# 
# The raw Pokémon Trading Card Game dataset contains valuable information, but many fields require preprocessing before they can be used by an AI battle agent.
# 
# In this notebook we will:
# 
# - Clean missing values
# - Normalize important features
# - Convert text into machine-readable values
# - Engineer new battle-related features
# - Produce an AI-ready dataset
# 
# The resulting dataset will serve as the foundation for deck construction, battle simulation, and strategic decision-making.

# In[1]:


from pathlib import Path

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 200)

PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

print("Project Root:", PROJECT_ROOT)
print("Raw Data:", DATA_DIR)
print("Processed Data:", PROCESSED_DIR)


# ## Step 1 – Load the English Dataset
# 
# The English dataset will be used as the primary source for feature engineering.
# 
# The Japanese dataset contains the same logical information and can later be processed using the same pipeline.

# In[2]:


cards = pd.read_csv(DATA_DIR / "EN_Card_Data.csv")

cards.shape


# ## Step 2 – Inspect the Dataset
# 
# Before modifying the data, we verify its structure and inspect the first few records.

# In[3]:


cards.head()


# In[4]:


cards.info()


# ## Observation 1
# 
# The dataset contains 2,022 records describing Pokémon cards.
# 
# Several columns contain missing values, which is expected because different card categories (Pokémon, Trainer, Energy) use different fields.
# 
# The next step is to clean and normalize these values for AI processing.

# ## Step 3 – Encode Pokémon Stage
# 
# Machine learning models work best with numerical values.
# 
# In this step, Pokémon evolution stages are converted into numeric values while Trainer and Energy cards receive a value of zero.

# In[5]:


stage_map = {
    "Basic Pokémon": 0,
    "Stage 1 Pokémon": 1,
    "Stage 2 Pokémon": 2
}

cards["stage_numeric"] = (
    cards["Stage (Pokémon)/Type (Energy and Trainer)"]
    .map(stage_map)
    .fillna(0)
)


# In[6]:


cards[
    [
        "Stage (Pokémon)/Type (Energy and Trainer)",
        "stage_numeric"
    ]
].head(20)


# ## Step 4 – Create Binary Evolution Features
# 
# In addition to encoding Pokémon evolution as a numeric value, we create binary indicator features.
# 
# These features simplify rule-based AI logic and improve compatibility with many machine learning algorithms.

# In[8]:


cards["is_basic"] = (
    cards["Stage (Pokémon)/Type (Energy and Trainer)"]
    == "Basic Pokémon"
).astype(int)

cards["is_stage1"] = (
    cards["Stage (Pokémon)/Type (Energy and Trainer)"]
    == "Stage 1 Pokémon"
).astype(int)

cards["is_stage2"] = (
    cards["Stage (Pokémon)/Type (Energy and Trainer)"]
    == "Stage 2 Pokémon"
).astype(int)


# In[9]:


cards[
    [
        "Card Name",
        "Stage (Pokémon)/Type (Energy and Trainer)",
        "stage_numeric",
        "is_basic",
        "is_stage1",
        "is_stage2"
    ]
].sample(10, random_state=42)


# ## Observation 3
# 
# Three binary evolution features were successfully created.
# 
# These indicators provide an efficient representation of Pokémon evolution stages and simplify downstream AI decision-making.
# 
# A Pokémon card can belong to only one evolution stage, while Trainer and Energy cards receive zeros for all three features.

# ## Step 5 – Create HP Feature
# 
# HP is one of the most important battle-related features.
# 
# The AI can use HP to estimate survivability, tank potential, and whether a Pokémon can remain active after taking damage.

# In[11]:


cards["hp_numeric"] = cards["HP"].fillna(0)


# In[12]:


cards[
    [
        "Card Name",
        "Stage (Pokémon)/Type (Energy and Trainer)",
        "HP",
        "hp_numeric"
    ]
].sample(10, random_state=42)


# ## Observation 4
# 
# The HP field was converted into an AI-ready numeric feature.
# 
# Cards without HP, such as Trainer and Energy cards, receive a value of 0 because they do not participate as Pokémon in battle.

# ## Step 6 – Create Retreat Cost Feature
# 
# Retreat Cost affects how easily a Pokémon can move from the Active Spot to the Bench.
# 
# A lower retreat cost gives the AI more flexibility during battle.

# In[13]:


cards["retreat_numeric"] = cards["Retreat"].fillna(0)


# In[14]:


cards[
    [
        "Card Name",
        "Stage (Pokémon)/Type (Energy and Trainer)",
        "Retreat",
        "retreat_numeric"
    ]
].sample(10, random_state=42)


# ## Observation 5
# 
# Retreat Cost was converted into a numeric AI-ready feature.
# 
# Cards without retreat costs, such as Energy and Trainer cards, receive a value of 0 because they do not occupy the Active Spot.

# ## Step 7 – Create Damage Feature
# 
# Attack damage is one of the most important offensive features.
# 
# Because some attacks have special text instead of simple numbers, we first convert the Damage column into a numeric feature.

# In[15]:


cards["damage_numeric"] = pd.to_numeric(cards["Damage"], errors="coerce").fillna(0)


# In[16]:


cards[
    [
        "Card Name",
        "Move Name",
        "Damage",
        "damage_numeric"
    ]
].sample(10, random_state=42)


# ## Observation 6
# 
# Attack damage was converted into a numeric AI-ready feature.
# 
# Non-numeric or missing damage values are assigned 0 for now. These attacks may still be strategically important because their effects are described in the Effect Explanation column.

# # Step 8 – Create Energy Cost Feature
# 
# The Energy Cost of an attack strongly influences battle strategy.
# 
# Powerful attacks often require multiple Energy cards before they can be used.
# 
# To make this information usable by machine learning models, we convert the Cost field into a numeric feature representing the total number of Energy required.

# In[19]:


def count_energy_symbols(cost):
    if pd.isna(cost):
        return 0

    text = str(cost)

    typed_energy = text.count("{")
    colorless_energy = text.count("●")

    return typed_energy + colorless_energy

cards["energy_cost"] = cards["Cost"].apply(count_energy_symbols)


# In[20]:


cards[
    [
        "Card Name",
        "Move Name",
        "Cost",
        "energy_cost"
    ]
].sample(10, random_state=42)


# ## Observation 7
# 
# The Energy Cost column has been transformed into a numeric feature.
# 
# Each attack now records the total number of Energy symbols required to perform that attack.
# 
# Trainer and Energy cards naturally receive a value of 0 because they do not have attack costs.

# # Step 9 – Explore Pokémon Types
# 
# Pokémon Type is one of the most influential battle features.
# 
# Type determines strengths, weaknesses, resistances, and many strategic interactions during battle.
# 
# Before converting this feature into machine-readable values, we first inspect all unique Pokémon types present in the dataset.

# In[21]:


cards["Type"].value_counts(dropna=False)


# # Step 10 – Create Primary Pokémon Type
# 
# Most Pokémon have a single primary type represented by one symbol.
# 
# For AI training, we simplify the raw Type field into a cleaner feature.
# 
# Trainer and Energy cards receive the value `"None"` because they do not have Pokémon battle types.
# 
# This feature will later be converted into machine-learning friendly numerical values.

# In[22]:


cards["primary_type"] = (
    cards["Type"]
    .fillna("None")
    .replace({
        "{G}": "Grass",
        "{R}": "Fire",
        "{W}": "Water",
        "{L}": "Lightning",
        "{P}": "Psychic",
        "{F}": "Fighting",
        "{D}": "Darkness",
        "{M}": "Metal",
        "{C}": "Colorless",
        "竜": "Dragon",
        "{A}": "Ancient",
        "{A}{A}": "Ancient",
        "{Team Rocket}{Team Rocket}": "Team Rocket",
        "{C}{C}{C}": "Colorless"
    })
)


# In[23]:


cards[
    [
        "Card Name",
        "Type",
        "primary_type"
    ]
].sample(15, random_state=42)


# ## Observation 8
# 
# The raw Pokémon Type field has been successfully converted into a clean categorical feature.
# 
# Special game symbols have been translated into readable Pokémon types, while Trainer and Energy cards are assigned the value `"None"`.
# 
# This transformation improves readability and prepares the feature for machine learning encoding in later stages.

# # Step 11 – Encode Primary Type
# 
# Machine learning algorithms work more efficiently with numeric values than text.
# 
# The cleaned Pokémon types are converted into numeric identifiers while preserving their categorical meaning.
# 
# Cards without Pokémon types remain encoded as zero.

# In[24]:


type_mapping = {
    "None": 0,
    "Grass": 1,
    "Fire": 2,
    "Water": 3,
    "Lightning": 4,
    "Psychic": 5,
    "Fighting": 6,
    "Darkness": 7,
    "Metal": 8,
    "Colorless": 9,
    "Dragon": 10,
    "Ancient": 11,
    "Team Rocket": 12
}

cards["type_numeric"] = cards["primary_type"].map(type_mapping)


# In[25]:


cards[
    [
        "Card Name",
        "primary_type",
        "type_numeric"
    ]
].sample(15, random_state=42)


# ## Observation 9
# 
# The categorical Pokémon types have been encoded into numeric values.
# 
# This representation is more suitable for statistical analysis, traditional machine learning algorithms, and neural networks.
# 
# Cards that are not Pokémon remain encoded as zero.

# # Step 12 – Create Weakness Feature
# 
# Weakness is one of the most important strategic features in the Pokémon Trading Card Game.
# 
# It determines which attacking types deal increased damage to a Pokémon.
# 
# Trainer and Energy cards do not have weaknesses, so they receive the value `"None"`.
# 
# The cleaned weakness feature will later be converted into machine-learning friendly numeric values.

# In[26]:


cards["weakness_type"] = (
    cards["Weakness"]
    .fillna("None")
    .replace({
        "{G}": "Grass",
        "{R}": "Fire",
        "{W}": "Water",
        "{L}": "Lightning",
        "{P}": "Psychic",
        "{F}": "Fighting",
        "{D}": "Darkness",
        "{M}": "Metal",
        "{C}": "Colorless",
        "竜": "Dragon",
        "{A}": "Ancient",
        "{A}{A}": "Ancient",
        "{Team Rocket}{Team Rocket}": "Team Rocket",
        "{C}{C}{C}": "Colorless"
    })
)


# In[27]:


cards[
    [
        "Card Name",
        "Weakness",
        "weakness_type"
    ]
].sample(15, random_state=42)


# ## Observation 10
# 
# The Weakness column has been transformed into readable categorical values.
# 
# Pokémon weaknesses are now represented using descriptive type names instead of game symbols.
# 
# Cards without weaknesses are assigned `"None"`.
# 
# This feature will later allow the AI to reason about battle type advantages.

# # Step 13 – Encode Weakness
# 
# To prepare the Weakness feature for machine learning, each weakness category is converted into a numeric identifier.
# 
# Using the same encoding as Pokémon types ensures consistency across the dataset.

# In[28]:


cards["weakness_numeric"] = (
    cards["weakness_type"]
    .map(type_mapping)
)


# In[29]:


cards[
    [
        "Card Name",
        "weakness_type",
        "weakness_numeric"
    ]
].sample(15, random_state=42)


# ## Observation 11
# 
# The Weakness feature has been encoded into numerical values using the same mapping as Pokémon types.
# 
# This consistency simplifies downstream machine learning and allows battle relationships between attacking and defending types to be modeled efficiently.

# # Step 14 – Create Resistance Feature
# 
# Resistance represents defensive advantages during battle.
# 
# A Pokémon with Resistance receives reduced damage from attacks of certain types.
# 
# Trainer and Energy cards do not have resistances, so they receive the value `"None"`.
# 
# This cleaned feature will later be encoded into machine-learning friendly numeric values.

# In[30]:


cards["resistance_type"] = (
    cards["Resistance (Type)"]
    .fillna("None")
    .replace({
        "{G}": "Grass",
        "{R}": "Fire",
        "{W}": "Water",
        "{L}": "Lightning",
        "{P}": "Psychic",
        "{F}": "Fighting",
        "{D}": "Darkness",
        "{M}": "Metal",
        "{C}": "Colorless",
        "竜": "Dragon",
        "{A}": "Ancient",
        "{A}{A}": "Ancient",
        "{Team Rocket}{Team Rocket}": "Team Rocket",
        "{C}{C}{C}": "Colorless"
    })
)


# In[31]:


cards[
    [
        "Card Name",
        "Resistance (Type)",
        "resistance_type"
    ]
].sample(15, random_state=42)


# ## Observation 12
# 
# The Resistance column has been transformed into readable categorical values.
# 
# Game symbols have been replaced with descriptive Pokémon types, while Trainer and Energy cards are assigned `"None"`.
# 
# This feature captures an important defensive mechanic that will later improve battle predictions.

# # Step 15 – Encode Resistance
# 
# To prepare Resistance for machine learning, each resistance type is mapped to the same numeric encoding used for Pokémon Type and Weakness.
# 
# Using one consistent encoding simplifies future feature engineering.

# In[32]:


cards["resistance_numeric"] = (
    cards["resistance_type"]
    .map(type_mapping)
)


# In[33]:


cards[
    [
        "Card Name",
        "resistance_type",
        "resistance_numeric"
    ]
].sample(15, random_state=42)


# ## Observation 13
# 
# Resistance has been encoded into numeric values using the same mapping as Pokémon Type and Weakness.
# 
# This consistent encoding allows the AI to compare attacking types against both defensive weaknesses and resistances during battle simulations.

# # Step 16 – Create Attack Power Score
# 
# Raw damage alone does not fully describe the effectiveness of an attack.
# 
# An attack requiring fewer Energy cards is generally more efficient than one requiring many Energy cards.
# 
# To capture this relationship, we create an Attack Power Score by dividing damage by the required Energy Cost.
# 
# A minimum Energy Cost of 1 is used to avoid division by zero.

# In[34]:


cards["attack_power_score"] = (
    cards["damage_numeric"] /
    cards["energy_cost"].replace(0, 1)
)


# In[35]:


cards[
[
    "Card Name",
    "Move Name",
    "damage_numeric",
    "energy_cost",
    "attack_power_score"
]
].sample(15, random_state=42)


# ## Observation 14
# 
# Attack Power Score measures the offensive efficiency of an attack.
# 
# Higher values indicate that more damage is produced for each Energy required.
# 
# This feature provides a better representation of offensive strength than raw damage alone.

# # Step 17 – Create Tank Score
# 
# Tank Score estimates how durable a Pokémon is during battle.
# 
# The score combines Hit Points with Retreat Cost.
# 
# Pokémon with higher HP and lower Retreat Costs receive better Tank Scores.

# In[36]:


cards["tank_score"] = (
    cards["hp_numeric"] /
    (cards["retreat_numeric"] + 1)
)


# In[37]:


cards[
[
    "Card Name",
    "hp_numeric",
    "retreat_numeric",
    "tank_score"
]
].sample(15, random_state=42)


# ## Observation 15
# 
# Tank Score estimates a Pokémon's ability to remain active during battle.
# 
# High HP combined with low Retreat Cost results in larger Tank Scores.
# 
# This feature captures defensive efficiency better than HP alone.

# # Step 18 – Create Mobility Score
# 
# Mobility measures how easily a Pokémon can retreat from battle.
# 
# Lower Retreat Costs produce higher Mobility Scores.
# 
# This feature helps the AI estimate tactical flexibility.

# In[38]:


cards["mobility_score"] = (
    1 /
    (cards["retreat_numeric"] + 1)
)


# In[39]:


cards[
[
    "Card Name",
    "retreat_numeric",
    "mobility_score"
]
].sample(15, random_state=42)


# ## Observation 16
# 
# Mobility Score represents how easily a Pokémon can leave the Active Spot.
# 
# Pokémon with no Retreat Cost receive the highest Mobility Scores.
# 
# This feature captures tactical flexibility during battle.

# # Step 19 – Create Evolution Score
# 
# Evolution stage affects card strategy.
# 
# Basic Pokémon are easier to play, while Stage 1 and Stage 2 Pokémon may offer stronger attacks or higher HP but require setup.
# 
# This feature gives the AI a simple numerical way to understand evolution progression.

# In[40]:


cards["evolution_score"] = cards["stage_numeric"]


# In[41]:


cards[
    [
        "Card Name",
        "Stage (Pokémon)/Type (Energy and Trainer)",
        "stage_numeric",
        "evolution_score"
    ]
].sample(15, random_state=42)


# ## Observation 17
# 
# Evolution Score represents how far a Pokémon is in its evolution line.
# 
# This helps the AI distinguish between quick setup cards and more developed Pokémon.

# # Step 20 – Create Rule Box Feature
# 
# Some Pokémon have special rule boxes, such as Pokémon ex.
# 
# These cards are often more powerful but may carry strategic risks.
# 
# This feature identifies whether a card contains rule text.

# In[43]:


cards["has_rule_box"] = cards["Rule"].notna().astype(int)


# In[44]:


cards[
    [
        "Card Name",
        "Rule",
        "has_rule_box"
    ]
].sample(15, random_state=42)


# ## Observation 18
# 
# The Rule field was converted into a binary feature.
# 
# Cards with rule text receive a value of 1, while cards without rule text receive 0.

# # Step 21 – Create Overall Battle Score
# 
# The Overall Battle Score combines several engineered features into one simple strategic rating.
# 
# This score is not a final model, but it gives the AI a useful starting point for comparing cards.

# In[45]:


cards["overall_battle_score"] = (
    cards["attack_power_score"]
    + cards["tank_score"]
    + cards["mobility_score"]
    + cards["evolution_score"]
    + cards["has_rule_box"]
)


# In[46]:


cards[
    [
        "Card Name",
        "Move Name",
        "attack_power_score",
        "tank_score",
        "mobility_score",
        "evolution_score",
        "has_rule_box",
        "overall_battle_score"
    ]
].sample(15, random_state=42)


# ## Observation 19
# 
# The Overall Battle Score combines offensive, defensive, mobility, evolution, and rule-box information.
# 
# This feature provides a first-pass strategic rating that can later be refined through simulation and evaluation.

# # Step 22 – Save the AI-Ready Dataset
# 
# After cleaning the dataset and engineering new features, we save the processed dataset.
# 
# This dataset will be used by future notebooks for:
# 
# - Deck construction
# - Card database creation
# - Battle simulation
# - AI decision making
# - Strategy evaluation

# In[47]:


output_file = PROCESSED_DIR / "pokemon_cards_processed.csv"

cards.to_csv(
    output_file,
    index=False
)

print("Processed dataset saved successfully!")
print(output_file)


# In[ ]:




