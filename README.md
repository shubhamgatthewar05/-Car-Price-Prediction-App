# Car Price Prediction App

This project is a Streamlit-based web application for predicting car prices. The app uses machine learning models to estimate the price of a car based on various input features.

## Features

- **Car Manufacturer**: Select from various car manufacturers.
- **Car Model**: Choose the specific model of the car.
- **Car Category**: Specify the category of the car (e.g., Sedan, SUV).
- **Leather Interior**: Choose if the car has a leather interior.
- **Fuel Type**: Select the type of fuel used (e.g., Petrol, Diesel).
- **Gear Box Type**: Choose the gear box type (e.g., Automatic, Manual).
- **Drive Wheels**: Specify the drive wheels (e.g., Front, Rear).
- **Number of Doors**: Choose the number of doors.
- **Levy, Mileage, Engine Volume, Cylinders, Airbags**: Provide numeric inputs for these features.

## Technologies Used

- **Streamlit**: For building the web application.
- **XGBoost**: For training the XGBoost model.
- **Scikit-Learn**: For preprocessing and Decision Tree modeling.
- **Pandas**: For data manipulation.
- **Numpy**: For numerical operations.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/car-price-prediction-app.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd car-price-prediction-app
    ```

3. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the App

To start the Streamlit app, use the following command:

```bash
streamlit run app.py
