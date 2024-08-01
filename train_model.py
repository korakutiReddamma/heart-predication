import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_data():
    data = pd.read_csv('heart.csv')  # Ensure the CSV file is in the correct path
    return data

def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_evaluate_model(model, params, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return best_model, accuracy

def train_models():
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=2000), {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }),
        'Random Forest': (RandomForestClassifier(), {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30]
        }),
        'Support Vector Machine': (SVC(), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9]
        }),
        'XGBoost': (XGBClassifier(eval_metric='mlogloss'), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        })
    }

    best_models = {}
    total_accuracy = 0
    for model_name, (model, params) in models.items():
        print(f"Training {model_name}...")
        best_model, accuracy = train_evaluate_model(model, params, X_train, y_train, X_test, y_test)
        best_models[model_name] = (best_model, accuracy)
        total_accuracy += accuracy
        print(f"Best {model_name} accuracy: {accuracy}")

    best_model_name = max(best_models, key=lambda k: best_models[k][1])
    best_model, best_accuracy = best_models[best_model_name]

    joblib.dump(best_models, 'model_accuracies.pkl')
    joblib.dump(best_model, 'best_model.pkl')
    
    return best_models, total_accuracy, best_model_name, best_accuracy

def get_accuracies():
    best_models, total_accuracy, best_model_name, best_accuracy = train_models()
    accuracies = {name: acc for name, (_, acc) in best_models.items()}
    return accuracies, total_accuracy, best_model_name, best_accuracy

if __name__ == "__main__":
    train_models()
