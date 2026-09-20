import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class StudentSuccessPredictor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = ['age', 'year_of_study', 'cgpa', 'attendance',
                               'hours_studied', 'previous_gpa', 'projects_completed',
                               'certifications', 'online_courses', 'internship_experience',
                               'study_consistency']

    def load_data(self):
        np.random.seed(42)
        n_students = 500
        data = {
            'student_id': [f"S{i+1:03d}" for i in range(n_students)],
            'age': np.random.randint(18, 26, n_students),
            'year_of_study': np.random.randint(1, 5, n_students),
            'cgpa': np.round(np.random.uniform(4.0, 9.5, n_students), 2),
            'attendance': np.round(np.random.uniform(50, 100, n_students), 1),
            'hours_studied': np.round(np.random.uniform(1, 10, n_students), 1),
            'previous_gpa': np.round(np.random.uniform(4.0, 9.0, n_students), 2),
            'internship_experience': np.random.choice(['Yes', 'No'], n_students, p=[0.4, 0.6]),
            'projects_completed': np.random.randint(0, 10, n_students),
            'certifications': np.random.randint(0, 5, n_students),
            'online_courses': np.random.randint(0, 8, n_students),
            'study_consistency': np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.3, 0.4, 0.3]),
        }
        df = pd.DataFrame(data)
        # Calculate success score correctly
        df['success_score'] = (
            df['cgpa']*0.25 + (df['attendance']/100)*1.5 + (df['hours_studied']/10)*1.5 +
            (df['projects_completed']/10)*1 + (df['certifications']/5)*1 +
            (df['online_courses']/8)*1 + (df['previous_gpa']/9)*1
        )
        df['success'] = (df['success_score'] >= 5.5).astype(int)
        self.data = df
        df.to_csv('students.csv', index=False)
        return df

    def preprocess_data(self):
        df = self.data.copy()
        le_map = {'No':0, 'Yes':1, 'Low':0, 'Medium':1, 'High':2}
        for col in ['internship_experience', 'study_consistency']:
            df[col] = df[col].map(le_map) if col == 'internship_experience' else df[col].map({'Low':0,'Medium':1,'High':2})

        X = df[self.feature_columns]
        y = df['success']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def train_models(self):
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression()
        }
        results = {}
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }
            self.models[name] = model
        self.best_model = self.models['Random Forest']
        return results

    def predict_success(self, student_data):
        features = np.array([student_data[col] for col in self.feature_columns]).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prob = self.best_model.predict_proba(features_scaled)[0][1]
        return {
            'success': bool(prob > 0.5),
            'probability': round(prob*100, 1),
            'risk_level': "Low" if prob > 0.7 else "Medium" if prob > 0.4 else "High"
        }