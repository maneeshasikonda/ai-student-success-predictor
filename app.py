from flask import Flask, render_template, request, jsonify
from predictor import StudentSuccessPredictor

app = Flask(__name__)
predictor = StudentSuccessPredictor()
predictor.load_data()
predictor.preprocess_data()
results = predictor.train_models()

print("✅ Model Trained!", results)

@app.route('/')
def home():
    return render_template('index.html', results=results, total_students=len(predictor.data))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert Yes/No to 1/0
    data['internship_experience'] = 1 if data['internship_experience'] == 'Yes' else 0
    data['study_consistency'] = {'Low':0,'Medium':1,'High':2}[data['study_consistency']]
    pred = predictor.predict_success(data)
    return jsonify(pred)

if __name__ == '__main__':
    app.run(debug=True)