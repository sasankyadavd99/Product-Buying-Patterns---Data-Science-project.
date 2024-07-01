
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations

# Load data
def load_data(file_info):
    dataframes = []
    for path, channel in file_info:
        df = pd.read_csv(path, low_memory=False)
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['Year'] = df['order_date'].dt.year
        df['Sales_Channel'] = channel
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['PRH4'].fillna('missing', inplace=True)
    combined_df['customer_group'].fillna('missing', inplace=True)
    return combined_df

# Load trained Random Forest model and preprocessor without caching
def load_rf_model():
    model = joblib.load('random_forest_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

# Apriori Algorithm
def apriori(basket, min_support, max_length=5):
    transactions = basket.apply(lambda x: x.index[x > 0].tolist(), axis=1).tolist()
    itemsets = {}
    
    def get_itemsets(transactions, length):
        return (frozenset(comb) for transaction in transactions for comb in combinations(transaction, length))

    def get_frequent_itemsets(transactions, itemsets, min_support):
        itemset_counts = {}
        for transaction in transactions:
            for itemset in itemsets:
                if itemset.issubset(transaction):
                    if itemset in itemset_counts:
                        itemset_counts[itemset] += 1
                    else:
                        itemset_counts[itemset] = 1
        num_transactions = len(transactions)
        return {itemset: count / num_transactions for itemset, count in itemset_counts.items() if count / num_transactions >= min_support}

    length = 1
    frequent_itemsets = get_frequent_itemsets(transactions, get_itemsets(transactions, length), min_support)
    all_frequent_itemsets = frequent_itemsets.copy()

    while frequent_itemsets and length < max_length:
        length += 1
        candidate_itemsets = get_itemsets(transactions, length)
        frequent_itemsets = get_frequent_itemsets(transactions, candidate_itemsets, min_support)
        all_frequent_itemsets.update(frequent_itemsets)

    return all_frequent_itemsets

# Generate Association Rules
def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for consequence in itemset:
                antecedent = itemset - frozenset([consequence])
                support = frequent_itemsets[itemset]
                confidence = support / frequent_itemsets[antecedent]
                if confidence >= min_confidence:
                    rules.append((antecedent, consequence, support, confidence))
    return rules

file_info = [
    ('/Users/SASANKYADAV1/Desktop/CAPSTONE PROJECT/Sponsor Data/2022_eShop.csv','eShop'),
    ('/Users/SASANKYADAV1/Desktop/CAPSTONE PROJECT/Sponsor Data/2022_non-eShop.csv', 'non-eShop'),
    ('/Users/SASANKYADAV1/Desktop/CAPSTONE PROJECT/Sponsor Data/2023_eShop.csv', 'eShop'),
    ('/Users/SASANKYADAV1/Desktop/CAPSTONE PROJECT/Sponsor Data/2023_non-eShop.csv', 'non-eShop'),
    ('/Users/SASANKYADAV1/Desktop/CAPSTONE PROJECT/Sponsor Data/2024_eShop.csv', 'eShop'),
    ('/Users/SASANKYADAV1/Desktop/CAPSTONE PROJECT/Sponsor Data/2024_non-eShop.csv', 'non-eShop')
]

# Load and prepare data
combined_df = load_data(file_info)
basket = (combined_df
          .groupby(['customer_group', 'PRH4'])['order_number']
          .count().unstack().reset_index().fillna(0)
          .set_index('customer_group'))
basket = basket.applymap(lambda x: 1 if x > 0 else 0)
frequent_itemsets = apriori(basket, min_support=0.01, max_length=5)
association_rules = generate_rules(frequent_itemsets, min_confidence=0.6)

# Load the model and preprocessor
rf, preprocessor = load_rf_model()

# Streamlit UI
st.title("Customer Segmentation and Product Prediction")
st.write("Select customer group, sales channel, and PRH1 category to predict PRH4 combinations.")

customer_groups = combined_df['customer_group'].unique()
prh1_categories = combined_df['PRH1'].unique()
sales_channels = combined_df['Sales_Channel'].unique()

customer_group = st.selectbox("Customer Group", customer_groups)
sales_channel = st.selectbox("Sales Channel", sales_channels)
prh1 = st.selectbox("PRH1 Category", prh1_categories)

if st.button("Predict PRH4 Combinations"):
    input_data = pd.DataFrame([[customer_group, sales_channel, prh1]], columns=['customer_group', 'Sales_Channel', 'PRH1'])
    input_data_prepared = preprocessor.transform(input_data)
    prh4_prediction = rf.predict(input_data_prepared)
    st.write(f"Predicted PRH4 Category: {prh4_prediction[0]}")

st.write("Association Rules for PRH4 Combinations:")
for rule in association_rules:
    st.write(f"Suggested PRH4 Category: {rule[1]}")
