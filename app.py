import os
import warnings
warnings.filterwarnings('ignore')

# numpy এর পুরনো ভার্সন ব্যবহার করতে বাধ্য করুন
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# মডেল লোড করার আগে numpy চেক করুন
print(f"NumPy version: {np.__version__}")

try:
    with open('student_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ মডেল সফলভাবে লোড হয়েছে")
except Exception as e:
    print(f"❌ মডেল লোড করতে সমস্যা: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        data = request.get_json()
        final_exam = float(data.get('final_exam', 0))
        midterm = float(data.get('midterm', 0))
        assignment = float(data.get('assignment', 0))
        
        # প্রেডিকশন
        prediction = model.predict([[final_exam, midterm, assignment]])[0]
        
        # গ্রেড নির্ধারণ
        if prediction >= 90:
            grade = "A"
        elif prediction >= 80:
            grade = "B"
        elif prediction >= 70:
            grade = "C"
        elif prediction >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return jsonify({
            'success': True,
            'predicted_score': round(prediction, 2),
            'grade': grade
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Student Performance API is running!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
