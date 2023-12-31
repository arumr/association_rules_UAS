import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from efficient_apriori import apriori


df = pd.read_csv('Market_Basket_Optimisation.csv',
                   header=None,
                   names=[f"item_{idx}" for idx in range(1, 21)]
)
df.replace(" asparagus", "asparagus", inplace=True)
basket_sizes = df.notna().apply(sum, axis=1)
baskets = [tuple(row.dropna()) for _, row in df[basket_sizes > 1].iterrows()]
item_sets, association_rules = apriori(baskets, min_support=0.01, min_confidence=0.3)
# print(df.head())


st.title("Groceries dataset Menggunakan Apriori ")


def user_input_features():
    item = st.selectbox("itemDescription",['asparagus','almonds','antioxydant juice','avocado','babies food','bacon',
 'barbecue sauce','black tea','blueberries','body spray','bramble',
 'brownies','bug spray', 'burger sauce', 'burgers' ,'butter', 'cake',
 'candy bars', 'carrots', 'cauliflower', 'cereals', 'champagne', 'chicken',
 'chili', 'chocolate', 'chocolate bread', 'chutney', 'cider',
 'clothes accessories', 'cookies', 'cooking oil', 'corn', 'cottage cheese',
 'cream', 'dessert wine', 'eggplant', 'eggs', 'energy bar', 'energy drink',
 'escalope', 'extra dark chocolate', 'flax seed', 'french fries',
 'french wine', 'fresh bread', 'fresh tuna', 'fromage blanc',
 'frozen smoothie', 'frozen vegetables', 'gluten free bar', 'grated cheese',
 'green beans', 'green grapes', 'green tea', 'ground beef', 'gums', 'ham',
 'hand protein bar', 'herb & pepper', 'honey', 'hot dogs', 'ketchup',
 'light cream', 'light mayo', 'low fat yogurt', 'magazines', 'mashed potato',
 'mayonnaise', 'meatballs', 'melons', 'milk', 'mineral water', 'mint',
 'mint green tea', 'muffins', 'mushroom cream sauce', 'napkins', 'nonfat milk',
 'oatmeal', 'oil', 'olive oil', 'pancakes', 'parmesan cheese', 'pasta', 'pepper',
 'pet food', 'pickles', 'protein bar', 'red wine', 'rice', 'salad', 'salmon',
 'salt', 'sandwich', 'shallot', 'shampoo', 'shrimp', 'soda', 'soup', 'spaghetti',
 'sparkling water', 'spinach', 'strawberries', 'strong cheese', 'tea',
 'tomato juice', 'tomato sauce', 'tomatoes', 'toothpaste', 'turkey',
 'vegetables mix', 'water spray', 'white wine', 'whole weat flour',
 'whole wheat pasta', 'whole wheat rice', 'yams', 'yogurt cake', 'zucchini'])
    
    return item

item = user_input_features()

def convert_values(value):
    if value <= 0:
        return 0
    elif value >=1:
        return 1


def return_item_df(data):
    target_data = data
    recommended_items = []

    one_to_one_rules = filter(
        lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, association_rules
    )

    for rule in sorted(one_to_one_rules, key=lambda rule: rule.lift):
        if target_data in rule.lhs:
            
            # Extract the recommended item ('rhs')
            recommended_item = [item for item in rule.rhs][0]
            
            recommended_items.append(recommended_item)

    return recommended_items


# if type(data) != type("No Result!"):
# print(return_item_df())
st.markdown("Hasil Rekomendasi :")
st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_df(item)}** secara bersamaan")

