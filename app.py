from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import mysql.connector
import secrets
from functools import wraps
import train_model
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
secret_key = secrets.token_hex(16)
app.secret_key = secret_key

# Load the model
model = joblib.load('best_model.pkl')

# In-memory storage for users and prediction data
users = {}
prediction_data = []
model_accuracies_dict = {}  # Renamed to avoid conflicts

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Reddy70@",
        database="heart_disease_db"
    )

@app.route('/')
def home():
    message = request.args.get('message', '')
    return render_template('home.html', message=message)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        gender = request.form['gender']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Check if passwords match
        if password != confirm_password:
            return render_template('register.html', message="Passwords do not match.")
        
        # Validate password complexity
        if len(password) < 8 or not any(char.isupper() for char in password) or not any(char.isdigit() for char in password) or not any(char in "@$!%*?&" for char in password):
            return render_template('register.html', message="Password must contain at least one uppercase letter, one number, one special character, and be at least 8 characters long.")
        
        # Check if the username is already taken
        if username in users:
            return render_template('register.html', message="Username already exists.")
        
        # Add the new user to the users dictionary
        users[username] = {
            'password': password,
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'gender': gender
        }
        
        session['username'] = username
        return redirect(url_for('predict'))

    return render_template('register.html')


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('register'))
        return f(*args, **kwargs)
    return wrap

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])
        except ValueError as e:
            print(f"ValueError: {e}")
            return render_template('predict.html', result_message="Invalid input data")

        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Create a DataFrame with feature names
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_df = pd.DataFrame(input_data, columns=feature_names)

        # Debugging: Print input data
        print(f"Input DataFrame: {input_df}")

        try:
            prediction = model.predict(input_df)

            # Debugging: Print prediction result
            print(f"Prediction result: {prediction[0]}")

            # Store the prediction data in MySQL database
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO predictions (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, 'Positive' if prediction[0] == 1 else 'Negative'))
            connection.commit()
            cursor.close()
            connection.close()

            return render_template('result.html', prediction_result='Positive' if prediction[0] == 1 else 'Negative')

        except Exception as e:
            print(f"Error: {e}")
            return render_template('result.html', prediction_result='Error')

    return render_template('predict.html')

@app.route('/result')
@login_required
def result():
    result_message = request.args.get('result_message', '')
    return render_template('result.html', result_message=result_message)

@app.route('/deaths')
@login_required
def deaths():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM heart_disease_deaths")
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    google_stats = {
        'total_cases': '1,234,567',
        'total_deaths': '123,456',
        'recovered': '1,000,000'
    }
    
    return render_template('deaths.html', data=data, google_stats=google_stats)

@app.route('/activities')
@login_required
def activities():
    return render_template('activities.html')

@app.route('/data')
@login_required
def data():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)  # Use dictionary cursor for easier access to column names
    cursor.execute("SELECT * FROM predictions")
    prediction_data = cursor.fetchall()
    cursor.close()
    connection.close()

    return render_template('data.html', prediction_data=prediction_data)

@app.route('/model_accuracies')
@login_required
def model_accuracies():
    accuracies, total_accuracy, best_model_name, best_accuracy = train_model.get_accuracies()
    return render_template('model_accuracies.html', accuracies=accuracies, total_accuracy=total_accuracy, best_model_name=best_model_name, best_accuracy=best_accuracy)
# Add the missing route for project done
@app.route('/projectdone')
def project_done():
    return render_template('projectdone.html')

# Add the missing route for project guide
@app.route('/projectguid')
def project_guid():
    return render_template('projectguid.html')
if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')
