# FoodWise -- Food Waste Prediction System

FoodWise is a machine-learning-based prototype that estimates food waste
for a planned campus-mess meal. It uses meal details, expected
attendance, planned food quantity, and historical records to generate an
experimental pre-meal estimate.

> **Important:** The model was developed using synthetic data because
> real operational mess records were not available. Its real-world
> prediction accuracy has not been established, and it is not validated
> for operational use. Predictions are estimates, not measurements of
> discarded food.

## Problem Statement

Campus messes must decide how much food to prepare before a meal.
Attendance can differ from expectations, which may contribute to excess
preparation and food waste. FoodWise explores whether historical meal
and attendance patterns can help estimate potential waste before food is
served.

## How It Works

1.  The user selects a meal and meal date.
2.  The user selects a menu that has an earlier record for that meal.
3.  The user enters expected attendance and planned food quantity.
4.  FoodWise retrieves historical information dated before the selected
    meal date.
5.  The saved machine-learning model generates an experimental
    food-waste estimate.
6.  The app displays the estimate and provides transparency into the
    historical source records and model inputs.

Historical features include previous attendance for the same menu,
previous waste for the meal category, and previous waste for the same
menu. Only earlier records are used for the historical lookup.

## Machine Learning

The training notebook evaluates regression models, including:

-   Linear Regression
-   Decision Tree Regressor
-   Random Forest Regressor
-   XGBoost Regressor

The model is selected using validation results. A separate later test
period is retained for final performance reporting. The saved model used
by the Streamlit application is `food_waste_model_updated.pkl`.

The target variable is `Waste kg`. The model uses pre-meal inputs;
actual attendance is not entered as a prediction-time feature.

## Technology Stack

-   **Python** -- application and model development
-   **Pandas** -- dataset handling and historical data processing
-   **NumPy** -- numerical operations
-   **Scikit-learn** -- preprocessing, regression pipelines, and
    evaluation
-   **XGBoost** -- additional regression model evaluated in the notebook
-   **Joblib** -- saving and loading the trained model
-   **Streamlit** -- interactive web application
-   **Plotly** -- analytics visualizations
-   **OpenPyXL** -- Excel workbook support

## Project Structure

``` text
Food Wise/
├── app.py
├── history.py
├── food_waste_model_updated.pkl
├── requirements.txt
├── test_updated_model.py
├── FoodWise_Model_Training_Updated.ipynb
├── FoodWise_Model_Training_Updated_XGBoost.ipynb
├── data/
│   └── FoodWise_Updated_Dataset.xlsx
└── .streamlit/
    └── config.toml
```

The exact files in the repository may vary as the project is updated.
The application expects the updated Excel dataset in the `data` folder,
with a worksheet named `Mess Data`.

## Setup and Run

### 1. Clone the repository

``` bash
git clone https://github.com/shauryagoel1407-droid/Food-Wise.git
cd Food-Wise
```

### 2. Create and activate a virtual environment (recommended)

On macOS or Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the application

``` bash
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in a
browser to use FoodWise.

## Dataset

The application reads `data/FoodWise_Updated_Dataset.xlsx`, specifically
the `Mess Data` worksheet. The dataset contains historical meal records
used by the application for historical lookups and analytics.

The dataset used for this prototype is synthetic. It should not be
interpreted as actual measurements from a university mess.

## Testing

Run the model and application checks with:

``` bash
python test_updated_model.py
```

The test script checks the updated model and related data/history
behavior. Make sure the required dataset and model files are present
before running it.

## Limitations

-   The dataset is synthetic, not collected from real mess operations.
-   Real-world prediction accuracy has not been established.
-   Results depend on the quality and coverage of the historical
    records.
-   Historical features require earlier records for the selected meal
    and menu.
-   The model's output is an experimental estimate and should not be
    used as the sole basis for food-preparation decisions.

## Future Scope

-   Validate the system using real, authorized mess records.
-   Retrain and evaluate the model on a larger, representative dataset.
-   Compare predictions with actual measured food waste over time.
-   Improve monitoring and reporting after real-world validation.

## Project

**FoodWise -- Food Waste Prediction System**\
Developed as an academic machine-learning project at Bennett University.
