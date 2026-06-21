# Nutritional Meal Classifier

A Flask web application that classifies meals as **Healthy** or **Unhealthy** based on their nutritional profile, powered by a Random Forest ensemble model trained with SMOTE balancing.

## Features

- Classifies meals using 8 nutritional inputs
- Displays confidence score with an animated progress bar
- Input validation with user-friendly error messages
- Retains form values after submission errors

## Nutritional Inputs

| Field | Unit |
|---|---|
| Calories | kcal |
| Protein | g |
| Carbohydrates | g |
| Fat | g |
| Fiber | g |
| Sugar | g |
| Sodium | mg |
| Cholesterol | mg |

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd meal-classifier

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Running in Debug Mode

```bash
set FLASK_DEBUG=true    # Windows
python app.py
```

## Tech Stack

- **Backend:** Python, Flask
- **ML Model:** Scikit-learn Random Forest + SMOTE (imbalanced-learn)
- **Frontend:** Bootstrap 5, Font Awesome 6, Inter (Google Fonts)
- **Data:** Pandas, Joblib

## Model

The trained model (`healthy_meal_model.pkl`) was built using:
- Random Forest Classifier
- SMOTE oversampling to handle class imbalance
- Features: calories, protein, carbs, fat, fiber, sugar, sodium, cholesterol
