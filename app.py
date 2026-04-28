import streamlit as st
import pandas as pd
import numpy as np
from model import load_data, train_model
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------------------
# 📊 Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Nassau Candy Distributor - Nassau Candy Distributor.csv")

    # ✅ Convert to datetime (FIX ERROR HERE)
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

    # ✅ Remove bad rows
    df = df.dropna(subset=['Order Date', 'Ship Date'])

    # ✅ Now calculate lead time
    df['lead_time'] = (df['Ship Date'] - df['Order Date']).dt.days

    return df

df = load_data()
model = train_model(df)
st.write(df[['Order Date', 'Ship Date']].head())
# -------------------------------
# 🏭 Factory Coordinates
# -------------------------------
factories = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107)
}

# -------------------------------
# 📏 Create Distance (Approx using Region mapping)
# -------------------------------
# Since dataset has no lat/lon, simulate by region grouping
region_map = {
    "West": (34.05, -118.24),
    "East": (40.71, -74.00),
    "Central": (41.25, -95.93),
    "South": (29.76, -95.36)
}

def get_distance(factory, region):
    if region in region_map:
        return geodesic(factories[factory], region_map[region]).km
    return np.random.randint(100, 2000)

# Add distance column
df['distance'] = df.apply(lambda x: get_distance("Lot's O' Nuts", x['Region']), axis=1)

# -------------------------------
# 🤖 Model Training
# -------------------------------
features = ['distance', 'Units', 'Sales']
X = df[features]
y = df['lead_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# -------------------------------
# 🔁 Simulation Function
# -------------------------------
def simulate(row):
    results = []

    for factory in factories:
        dist = get_distance(factory, row['Region'])

        features = [[dist, row['Units'], row['Sales']]]
        pred = model.predict(features)[0]

        results.append({
            "Factory": factory,
            "Predicted Lead Time": round(pred, 2)
        })

    return sorted(results, key=lambda x: x['Predicted Lead Time'])

# -------------------------------
# 🏆 Recommendation
# -------------------------------
def recommend(row):
    sims = simulate(row)
    current = row['lead_time']

    for s in sims:
        s['Improvement'] = round(current - s['Predicted Lead Time'], 2)

    return sorted(sims, key=lambda x: x['Improvement'], reverse=True)

# -------------------------------
# 🌐 STREAMLIT UI
# -------------------------------
st.title("🏭 Factory Reallocation & Shipping Optimization")

st.markdown("### Select Order for Optimization")

order_id = st.selectbox("Order ID", df['Order ID'].unique())

row = df[df['Order ID'] == order_id].iloc[0]

# Show current info
st.write("### 📦 Current Order Details")
st.write({
    "Product": row['Product Name'],
    "Region": row['Region'],
    "Ship Mode": row['Ship Mode'],
    "Current Lead Time": row['lead_time']
})

# -------------------------------
# 🚀 Run Optimization
# -------------------------------
if st.button("Run Optimization"):

    results = recommend(row)
    result_df = pd.DataFrame(results)

    st.write("### 🏆 Recommended Factories")
    st.dataframe(result_df)

    # KPI
    old_time = row['lead_time']
    best_time = results[0]['Predicted Lead Time']

    reduction = ((old_time - best_time) / old_time) * 100

    st.success(f"🚀 Lead Time Reduction: {reduction:.2f}%")