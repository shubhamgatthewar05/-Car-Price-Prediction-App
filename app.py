import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_preprocess_data():
    df = pd.read_csv('car_price_prediction.csv')
    
  
    df = df.drop(columns=['ID'], errors='ignore')
    
 
    numeric_features = ['Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')  
    
 
    X = df.drop(columns=['Price'])
    y = df['Price']
    
 
    y = y.dropna()
    X = X.loc[y.index]

    categorical_features = ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Doors']
    
 
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])  
 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
   
    X_preprocessed = preprocessor.fit_transform(X)
    
    return X_preprocessed, y, preprocessor


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Train Decision Tree model
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    
    return xgb_model, dt_model

# Predict car price using the selected model
def predict_price(model, preprocessor, input_data):
    input_df = pd.DataFrame([input_data])
    input_df_preprocessed = preprocessor.transform(input_df)
    prediction = model.predict(input_df_preprocessed)
    return prediction[0]

# Streamlit app
def main():
    st.title('Car Price Prediction App')

    # Load and preprocess data
    X_preprocessed, y, preprocessor = load_and_preprocess_data()

    # Train models
    xgb_model, dt_model = train_models(X_preprocessed, y)

    st.sidebar.header('User Input')
    
    # Define sidebar inputs
    manufacturer = st.sidebar.selectbox('Manufacturer', ['Ford', 'BMW', 'Audi', 'Toyota'])
    model_input = st.sidebar.selectbox('Model', ['Focus', 'X5', 'A4', 'Corolla'])
    category = st.sidebar.selectbox('Category', ['Sedan', 'SUV', 'Hatchback'])
    leather_interior = st.sidebar.selectbox('Leather Interior', ['Yes', 'No'])
    fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric'])
    gear_box_type = st.sidebar.selectbox('Gear Box Type', ['Automatic', 'Manual'])
    drive_wheels = st.sidebar.selectbox('Drive Wheels', ['Front', 'Rear', 'All'])
    doors = st.sidebar.selectbox('Doors', [2, 4, 5])
    levy = st.sidebar.number_input('Levy', value=0.0)
    mileage = st.sidebar.number_input('Mileage', value=0.0)
    engine_volume = st.sidebar.number_input('Engine Volume', value=0.0)
    cylinders = st.sidebar.number_input('Cylinders', value=4)
    airbags = st.sidebar.number_input('Airbags', value=2)
    
    input_data = {
        'Manufacturer': manufacturer,
        'Model': model_input,
        'Category': category,
        'Leather interior': leather_interior,
        'Fuel type': fuel_type,
        'Gear box type': gear_box_type,
        'Drive wheels': drive_wheels,
        'Doors': doors,
        'Levy': levy,
        'Mileage': mileage,
        'Engine volume': engine_volume,
        'Cylinders': cylinders,
        'Airbags': airbags
    }

    st.sidebar.header('Select Model')
    model_choice = st.sidebar.selectbox('Choose Model', ['XGBoost', 'Decision Tree'])

    if st.sidebar.button('Predict'):
        if model_choice == 'XGBoost':
            price_prediction = predict_price(xgb_model, preprocessor, input_data)
        else:
            price_prediction = predict_price(dt_model, preprocessor, input_data)
        
        st.write(f"Predicted Car Price using {model_choice}: ${price_prediction:.2f}")

if __name__ == '__main__':
    main()
