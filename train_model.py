import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# 1. Dummy Dataset Creation (Since running in safe-mode)
data = {
    'Life_Expectancy': [75.5, 68.2, 81.3, 54.1, 72.8, 79.1, 62.4, 83.2, 59.8, 70.1],
    'Mean_Years_Schooling': [12.2, 8.5, 13.4, 4.2, 10.1, 12.8, 6.2, 13.9, 5.1, 9.4],
    'GNI_Per_Capita': [15000, 6200, 43000, 1200, 9800, 28000, 3100, 52000, 1800, 7500],
    'HDI_Score': [0.785, 0.642, 0.924, 0.385, 0.710, 0.865, 0.520, 0.945, 0.440, 0.675]
}

df = pd.DataFrame(data)

# 2. Splitting Features and Target
X = df[['Life_Expectancy', 'Mean_Years_Schooling', 'GNI_Per_Capita']]
y = df['HDI_Score']

# 3. Model Training using Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Saving the model to model.pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and model.pkl saved successfully!")
