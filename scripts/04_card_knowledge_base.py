#!/usr/bin/env python
# coding: utf-8

# # Notebook 04 – Card Knowledge Base
# 
# ## Objective
# 
# The feature-engineered dataset contains AI-ready numerical features, but the battle engine requires each card to be represented as an object rather than as a table row.
# 
# In this notebook we will:
# 
# - Load the processed Pokémon TCG dataset
# - Create Python classes for Pokémon, Trainer, and Energy cards
# - Build an Attack class to represent individual attacks
# - Construct a searchable Card Database
# - Verify that cards can be retrieved and inspected programmatically
# 
# The resulting Card Knowledge Base will become the foundation for the deck builder, battle simulator, and AI agent.

# # Step 1 – Load Processed Dataset
# 
# The feature engineering notebook produced an AI-ready dataset containing cleaned values and engineered features.
# 
# In this notebook, we load that processed dataset and use it as the foundation for constructing Pokémon card objects.
# 
# Using the processed data ensures that every card shares the same standardized representation required by the battle engine.

# In[1]:


from pathlib import Path
import pandas as pd

# Project directories
PROJECT_ROOT = Path.cwd()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

print("Project Root :", PROJECT_ROOT)
print("Processed Data:", PROCESSED_DIR)


# In[3]:


# Load processed dataset

cards = pd.read_csv(PROCESSED_DIR / "pokemon_cards_processed.csv")

print("Processed dataset loaded successfully.")


# In[5]:


cards.shape


# In[6]:


cards.head()


# ## Observation 1
# 
# The processed CSV dataset was successfully loaded.
# 
# This file will be used to build the Card Knowledge Base.

# # Step 2 – Inspect Engineered Features
# 
# The processed dataset contains additional features generated during preprocessing.
# 
# Before constructing Pokémon card objects, we verify that these engineered features are available.
# 
# These features will become the core attributes used by the AI battle engine.

# In[9]:


# Display every column

cards.columns.tolist()


# In[13]:


stage_type_col = "Stage (Pokémon)/Type (Energy and Trainer)"

cards["is_pokemon"] = cards[stage_type_col].isin(
    ["Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"]
).astype(int)

cards["is_trainer"] = cards[stage_type_col].isin(
    ["Item", "Supporter", "Pokémon Tool", "Stadium"]
).astype(int)

cards["is_energy"] = cards[stage_type_col].isin(
    ["Basic Energy", "Special Energy"]
).astype(int)

def assign_card_category(value):
    if value in ["Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"]:
        return "Pokémon"
    if value in ["Item", "Supporter", "Pokémon Tool", "Stadium"]:
        return "Trainer"
    if value in ["Basic Energy", "Special Energy"]:
        return "Energy"
    return "Other"

cards["card_category"] = cards[stage_type_col].apply(assign_card_category)

cards["has_attack"] = cards["Move Name"].notna().astype(int)

cards["has_ability"] = (
    cards["Effect Explanation"]
    .fillna("")
    .str.contains("Ability", case=False, regex=False)
).astype(int)

print("Missing features created in Notebook 04.")


# In[14]:


engineered_features = [
    "stage_numeric",
    "is_basic",
    "is_stage1",
    "is_stage2",
    "hp_numeric",
    "retreat_numeric",
    "damage_numeric",
    "energy_cost",
    "primary_type",
    "type_numeric",
    "weakness_type",
    "weakness_numeric",
    "resistance_type",
    "resistance_numeric",
    "attack_power_score",
    "tank_score",
    "mobility_score",
    "evolution_score",
    "has_rule_box",
    "is_pokemon",
    "is_trainer",
    "is_energy",
    "card_category",
    "has_attack",
    "has_ability"
]

cards[engineered_features].head()


# # Step 3 – Identify Unique Cards
# 
# The dataset contains one row per attack, so some cards appear more than once.
# 
# Before building card objects, we identify how many unique Card IDs exist.

# In[15]:


total_rows = len(cards)
unique_cards = cards["Card ID"].nunique()

print("Total rows:", total_rows)
print("Unique Card IDs:", unique_cards)
print("Extra attack rows:", total_rows - unique_cards)


# ## Observation 3
# 
# The processed dataset still uses one row per attack.
# 
# Card objects should therefore be grouped by Card ID so that each card can contain one or more attacks.

# # Step 4 – Count Attacks per Card
# 
# Many Pokémon cards contain multiple attacks.
# 
# Before creating card objects, we determine how many attacks belong to each Card ID.
# 
# This allows the AI to store every attack inside a single Pokémon card object.

# In[16]:


attack_counts = (
    cards.groupby("Card ID")["Move Name"]
    .count()
    .reset_index(name="number_of_attacks")
)

attack_counts.head(10)


# ## Observation 4
# 
# Each Card ID has an associated number of attacks.
# 
# Most Pokémon have one or two attacks.
# 
# Trainer and Energy cards usually have zero attacks.

# In[17]:


cards = cards.merge(
    attack_counts,
    on="Card ID",
    how="left"
)

cards[
    [
        "Card ID",
        "Card Name",
        "Move Name",
        "number_of_attacks"
    ]
].sample(10, random_state=42)


# ## Observation 5
# 
# Every row now stores the total number of attacks for its card.
# 
# This feature will simplify the construction of complete card objects.

# # Step 6 – Build the Initial Card Knowledge Base
# 
# The processed dataset contains one row per attack.
# 
# To prepare for battle simulation, we create one entry for each unique card.
# 
# Each card object will later store:
# 
# • Card information
# • HP
# • Pokémon Type
# • Weakness
# • Resistance
# • Retreat Cost
# • All attacks
# • Rules
# • Abilities

# In[18]:


card_base = (
    cards
    .drop_duplicates(subset="Card ID")
    .copy()
)

print("Unique cards:", len(card_base))

card_base.head()


# ## Observation 6
# 
# The Card Knowledge Base now contains one row for every unique Pokémon card.
# 
# The next step is to collect all attacks that belong to each card into a structured list.

# In[19]:


Pikachu = {
    "name": "Pikachu",
    "hp": 60,
    "type": "Lightning",
    "attacks": [
        {
            "name": "Thunder Jolt",
            "cost": 1,
            "damage": 30,
            "effect": "..."
        },
        {
            "name": "Volt Tackle",
            "cost": 3,
            "damage": 120,
            "effect": "..."
        }
    ]
}


# # Step 7 – Build Attack Lists
# 
# The processed dataset currently stores one attack per row.
# 
# However, a Pokémon card may have multiple attacks.
# 
# To create a true Card Knowledge Base, we collect every attack belonging to a card into a single list.
# 
# Each card object will later contain:
# 
# - Basic card information
# - HP
# - Type
# - Weakness
# - Resistance
# - Retreat Cost
# - A list of attacks
# - Rules
# - Ability information

# In[20]:


attack_data = (
    cards[
        [
            "Card ID",
            "Move Name",
            "Cost",
            "Damage",
            "Effect Explanation",
            "energy_cost",
            "damage_numeric"
        ]
    ]
    .copy()
)

attack_data.head()


# ## Observation 7
# 
# Each row now contains only attack-related information.
# 
# These rows will soon be grouped into complete attack lists for each Pokémon.

# In[21]:


attack_lists = (
    attack_data
    .groupby("Card ID")
    .apply(
        lambda df: df[
            [
                "Move Name",
                "Cost",
                "Damage",
                "Effect Explanation",
                "energy_cost",
                "damage_numeric"
            ]
        ].to_dict("records")
    )
    .reset_index(name="attacks")
)

attack_lists.head()


# In[22]:


[
    {
        "Move Name": "Thunder Jolt",
        "Cost": "{L}",
        "Damage": "30",
        "Effect Explanation": "...",
        "energy_cost": 1,
        "damage_numeric": 30
    },
    {
        "Move Name": "Volt Tackle",
        "Cost": "{L}{L}{C}",
        "Damage": "120",
        "Effect Explanation": "...",
        "energy_cost": 3,
        "damage_numeric": 120
    }
]


# ## Observation 8
# 
# Every Card ID now owns a complete list of attacks.
# 
# This converts the flat attack table into a structured representation suitable for AI reasoning.

# In[23]:


card_base = card_base.merge(
    attack_lists,
    on="Card ID",
    how="left"
)

card_base.head()


# In[24]:


example = card_base[
    card_base["Card Name"].str.contains(
        "Pikachu",
        case=False,
        na=False
    )
]

example[
    [
        "Card Name",
        "HP",
        "primary_type",
        "attacks"
    ]
]


# In[25]:


card_base.sample(1, random_state=42)[
    [
        "Card Name",
        "HP",
        "primary_type",
        "attacks"
    ]
]


# ## Observation 9
# 
# Each card now contains its complete attack list.
# 
# This is a major milestone because the Card Knowledge Base now stores one complete object per card rather than one row per attack.

# Pikachu = {
#     "HP": 60,
#     "Type": "Lightning",
#     "Attacks": [
#         Thunder Jolt,
#         Volt Tackle
#     ]
# }

# # Step 8 – Build Complete Card Objects
# 
# The Card Knowledge Base currently stores one row per card.
# 
# For battle simulation, we convert each row into a Python dictionary.
# 
# Each dictionary represents one complete Pokémon TCG card and contains:
# 
# • General card information
# • Battle statistics
# • Weaknesses
# • Resistances
# • Retreat Cost
# • Attack list
# • Rule Box
# • Engineered features
# 
# These objects will become the fundamental data structure used by the battle engine.

# In[27]:


card_objects = {}

for _, row in card_base.iterrows():

    card_objects[row["Card ID"]] = {

        "card_id": row["Card ID"],

        "name": row["Card Name"],

        "category": row["card_category"],

        "stage": row["Stage (Pokémon)/Type (Energy and Trainer)"],

        "hp": row["HP"],

        "type": row["primary_type"],

        "weakness": row["weakness_type"],

        "resistance": row["resistance_type"],

        "retreat": row["retreat_numeric"],

        "overall_score": row["overall_battle_score"],

        "attacks": row["attacks"],

        "rule": row["Rule"],

        "effect": row["Effect Explanation"]

    }

print("Total card objects:", len(card_objects))


# In[28]:


first_key = next(iter(card_objects))

card_objects[first_key]


# In[29]:


{
 'card_id': 1,

 'name': 'Bulbasaur',

 'category': 'Pokémon',

 'hp': 70,

 'type': 'Grass',

 'weakness': 'Fire',

 'retreat': 1,

 'overall_score': 83.2,

 'attacks': [

      {
         'Move Name':'Vine Whip',
         'damage_numeric':30,
         'energy_cost':2
      }

 ],

 'rule': None,

 'effect': None

}


# In[ ]:




