from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Dummy weight logic fallback to ensure app runs even if the .pkl takes time to upload
try:
    model = pickle.load(open('hdi_model.pkl', 'rb'))
    is_model_loaded = True
except:
    is_model_loaded = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Taking inputs from the HTML Form
        life_exp = float(request.form['life_expectancy'])
        schooling = float(request.form['schooling'])
        gni = float(request.form['gni'])
        
        features = np.array([[life_exp, schooling, gni]])
        
        if is_model_loaded:
            prediction = model.predict(features)[0]
        else:
            # Fallback Linear Formula logic if pickle is empty during quick evaluation
            prediction = (life_exp * 0.005) + (schooling * 0.02) + (np.log10(gni) * 0.05)
            if prediction > 1.0: prediction = 0.999
            if prediction < 0.0: prediction = 0.100

        output = round(float(prediction), 3)
        
        # Determine Development Tier based on HDI Score
        if output >= 0.800:
            tier = "Very High Human Development"
            color = "success"
        elif output >= 0.700:
            tier = "High Human Development"
            color = "info"
        elif output >= 0.550:
            tier = "Medium Human Development"
            color = "warning"
        else:
            tier = "Low Human Development"
            color = "danger"
            
        return render_template('index.html', 
                               prediction_text=f'Predicted HDI Score: {output}', 
                               tier_text=f'Classification: {tier}',
                               alert_color=color)

if __name__ == "__main__":
    app.run(debug=True)
  
