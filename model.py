import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------------------
# 📊 Load & Clean Data
# -------------------------------
def load_data():
    df = pd.read_csv("data/Nassau Candy Distributor - Nassau Candy Distributor.csv")

    # Convert to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

    # Drop invalid rows
    df = df.dropna(subset=['Order Date', 'Ship Date'])

    # Create lead time
    df['lead_time'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Remove unrealistic values
    df = df[(df['lead_time'] > 0) & (df['lead_time'] < 30)]

    # Profit margin
    df['profit_margin'] = df['Gross Profit'] / df['Sales']

    return df


# -------------------------------
# 🤖 Train Model
# -------------------------------
def train_model(df):
    # Features
    features = ['Units', 'Sales']

    X = df[features]
    y = df['lead_time']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model