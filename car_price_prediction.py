import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load dataset
df = pd.read_csv('ferrari_812_superfast_sample.csv')

# Feature Engineering
df['age'] = 2024 - df['year']
df['days_since_sale'] = df['days_since_sale']  # Already in days

# Create brand-specific average price features
df['brand_avg_price'] = df.groupby('brand')['price'].transform('mean')

# Create engine type similarity feature
df['engine_type_similarity'] = df.groupby('engine')['price'].transform('mean')

# Define features and target
features = ['age', 'engine', 'transmission', 'num_produced', 'mileage', 'engine_size', 'brand_avg_price', 'engine_type_similarity', 'days_since_sale']
target = 'price'

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'mileage', 'engine_size', 'num_produced', 'brand_avg_price', 'engine_type_similarity', 'days_since_sale']),
        ('cat', OneHotEncoder(), ['engine', 'transmission'])
    ])

X = df[features]
y = df[target]

# Normalize target variable
y = np.log1p(y)  # Log transformation to handle large values

X_preprocessed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

def predict_price(year, engine, transmission, num_produced, brand, engine_size, days_since_sale):
    # Calculate the age
    age = 2024 - year
    
    # Get average price for brand
    brand_avg_price = df[df['brand'] == brand]['price'].mean()
    
    # Get engine type similarity
    engine_type_similarity = df[df['engine'] == engine]['price'].mean()
    
    # Prepare input data with all features
    # Note: Features should match the order and number of columns used in preprocessing
    input_data = np.array([[age, engine_size, num_produced, brand_avg_price, engine_type_similarity, days_since_sale] + [engine, transmission]])
    
    # Ensure input data matches the expected format of the preprocessor
    expected_features = len(preprocessor.transformers_[0][1].get_feature_names_out()) + len(preprocessor.transformers_[1][1].get_feature_names_out())
    if len(input_data[0]) != expected_features:
        raise ValueError(f"Input data does not match the expected number of features for the preprocessor. Expected {expected_features}, got {len(input_data[0])}.")
    
    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)
    
    # Predict the price
    predicted_price = model.predict(input_data_preprocessed)[0, 0]
    
    # Inverse log transformation to get the original price
    return np.expm1(predicted_price)

def create_animation():
    fig, ax = plt.subplots()
    years = np.arange(df['year'].min(), 2026)
    prices = []

    # Aggregate features for a generic Ferrari 812 Superfast
    selected_brand = 'Ferrari'
    model_data = df[df['brand'] == selected_brand]
    
    if not model_data.empty:
        # Compute average features for the selected brand
        avg_features = model_data[['engine', 'transmission', 'num_produced', 'engine_size']].mode().iloc[0]
        avg_features = avg_features.to_dict()
        engine = avg_features['engine']
        transmission = avg_features['transmission']
        num_produced = avg_features['num_produced']
        engine_size = avg_features['engine_size']

        def update(year):
            days_since_sale = (pd.Timestamp('2024-01-01') - pd.Timestamp(f'{year}-01-01')).days
            # Predict price for a specific year
            price = predict_price(year, engine, transmission, num_produced, selected_brand, engine_size, days_since_sale)
            prices.append(price)
            ax.clear()
            ax.plot(years[:len(prices)], prices, color='blue')
            ax.set_xlabel('Year')
            ax.set_ylabel('Price')
            ax.set_title('Price Prediction for Ferrari 812 Superfast')
            return ax

        ani = animation.FuncAnimation(fig, update, frames=years, repeat=False, blit=False)
        plt.show()
    else:
        print(f"No data available for the selected brand: {selected_brand}")

create_animation()

def create_animation():
    fig, ax = plt.subplots()
    years = np.arange(df['year'].min(), 2026)
    prices = []

    # Aggregate features for a generic Ferrari 812 Superfast
    selected_brand = 'Ferrari'
    model_data = df[df['brand'] == selected_brand]
    
    if not model_data.empty:
        # Compute average features for the selected brand
        avg_features = model_data[['engine', 'transmission', 'num_produced', 'engine_size']].mode().iloc[0]
        avg_features = avg_features.to_dict()
        engine = avg_features['engine']
        transmission = avg_features['transmission']
        num_produced = avg_features['num_produced']
        engine_size = avg_features['engine_size']

        def update(year):
            days_since_sale = (pd.Timestamp('2024-01-01') - pd.Timestamp(f'{year}-01-01')).days
            # Predict price for a specific year
            price = predict_price(year, engine, transmission, num_produced, selected_brand, engine_size, days_since_sale)
            prices.append(price)
            ax.clear()
            ax.plot(years[:len(prices)], prices, color='blue')
            ax.set_xlabel('Year')
            ax.set_ylabel('Price')
            ax.set_title('Price Prediction for Ferrari 812 Superfast')
            return ax

        ani = animation.FuncAnimation(fig, update, frames=years, repeat=False, blit=False)
        plt.show()
    else:
        print(f"No data available for the selected brand: {selected_brand}")

create_animation()
