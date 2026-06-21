import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('healthy_meal_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    error = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form  # retain values on any error path

        try:
            features = {
                'calories':       float(request.form['calories']),
                'protein_g':      float(request.form['protein']),
                'carbs_g':        float(request.form['carbs']),
                'fat_g':          float(request.form['fat']),
                'fiber_g':        float(request.form['fiber']),
                'sugar_g':        float(request.form['sugar']),
                'sodium_mg':      float(request.form['sodium']),
                'cholesterol_mg': float(request.form['cholesterol'])
            }

            if any(v < 0 for v in features.values()):
                error = "All nutritional values must be 0 or greater."
            else:
                input_df = pd.DataFrame([features])
                ai_prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                confidence = round(max(probabilities) * 100, 1)
                prediction = "HEALTHY" if ai_prediction == 1 else "UNHEALTHY"

        except ValueError:
            error = "Please enter valid numeric values for all fields."
        except Exception as e:
            error = f"An unexpected error occurred: {e}"

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        error=error,
        form_data=form_data
    )

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug)
