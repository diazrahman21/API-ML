from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import datetime
import os
import sys
import traceback
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with more permissive settings for production
CORS(app, 
     origins=['*'],  # Allow all origins for now
     allow_headers=['Accept', 'Authorization', 'Content-Type', 'If-None-Match', 'cache-control', 'x-requested-with'],
     supports_credentials=False,  # Set to False when using wildcard
     methods=['GET', 'POST', 'OPTIONS'])

# Global variables for model and preprocessing objects
model = None
scaler = None
feature_info = None

def load_model_and_preprocessors():
    """Load model and preprocessing objects"""
    global model, scaler, feature_info
    
    try:
        print("🔍 Starting model loading process...")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Files in directory: {os.listdir('.')}")
        
        # Check if files exist
        model_files = ['my_best_model.h5', 'scaler.pkl', 'feature_info.pkl']
        missing_files = [f for f in model_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        print("✅ All required files found")
        
        # Load model with detailed error handling
        print("📁 Loading Keras model...")
        try:
            model = load_model('my_best_model.h5', compile=False)
            print(f"✅ Model loaded successfully - Type: {type(model)}")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False
        
        # Load scaler
        print("📁 Loading scaler...")
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler loaded successfully - Type: {type(scaler)}")
        except Exception as e:
            print(f"❌ Error loading scaler: {str(e)}")
            return False
        
        # Load feature info
        print("📁 Loading feature info...")
        try:
            with open('feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
            print(f"✅ Feature info loaded successfully - Type: {type(feature_info)}")
        except Exception as e:
            print(f"❌ Error loading feature info: {str(e)}")
            return False
        
        print("🎉 All components loaded successfully!")
        return True
    
    except Exception as e:
        print(f"❌ Unexpected error in load_model_and_preprocessors: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def convert_age_to_days(age_years):
    """Convert age from years to days"""
    return int(age_years * 365.25)

def convert_age_to_years(age_days):
    """Convert age from days to years"""
    return round(age_days / 365.25, 1)

# Load model and preprocessors on startup
print("🚀 Starting Cardiovascular Disease Prediction API...")
print(f"🐍 Python version: {sys.version}")
print(f"🧠 TensorFlow version: {tf.__version__}")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🌐 Port: {os.environ.get('PORT', 'Not set')}")

# Try to load models immediately with error handling
try:
    if load_model_and_preprocessors():
        print("✅ Initialization successful - All models loaded")
    else:
        print("⚠️ Initialization failed - Models will be loaded on first request")
except Exception as e:
    print(f"❌ Startup error: {str(e)}")
    print("⚠️ Will attempt to load models on first request")

# Add a simple health check that responds quickly
@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint for Railway health checks"""
    return jsonify({"status": "ok", "timestamp": datetime.datetime.now().isoformat()})

@app.route('/dataset-info', methods=['GET'])
def dataset_info():
    """Get detailed dataset information"""
    return jsonify({
        "title": "Cardiovascular Disease Dataset",
        "description": "Dataset for predicting cardiovascular disease based on medical examination data",
        "data_collection": "All dataset values were collected at the moment of medical examination",
        "target_variable": {
            "name": "cardio",
            "description": "Presence or absence of cardiovascular disease",
            "type": "binary",
            "values": {
                "0": "No cardiovascular disease",
                "1": "Cardiovascular disease present"
            }
        },
        "feature_types": {
            "objective": "Factual information",
            "examination": "Results of medical examination", 
            "subjective": "Information given by the patient"
        },
        "features": [
            {
                "name": "age",
                "description": "Age of patient",
                "type": "Objective Feature",
                "data_type": "int",
                "unit": "days",
                "note": "Age converted from years to days (years * 365.25)"
            },
            {
                "name": "height", 
                "description": "Height of patient",
                "type": "Objective Feature",
                "data_type": "int",
                "unit": "cm"
            },
            {
                "name": "weight",
                "description": "Weight of patient", 
                "type": "Objective Feature",
                "data_type": "float",
                "unit": "kg"
            },
            {
                "name": "gender",
                "description": "Gender of patient",
                "type": "Objective Feature", 
                "data_type": "categorical code",
                "values": {
                    "1": "Male",
                    "2": "Female"
                }
            },
            {
                "name": "ap_hi",
                "description": "Systolic blood pressure",
                "type": "Examination Feature",
                "data_type": "int",
                "unit": "mmHg"
            },
            {
                "name": "ap_lo", 
                "description": "Diastolic blood pressure",
                "type": "Examination Feature",
                "data_type": "int", 
                "unit": "mmHg"
            },
            {
                "name": "cholesterol",
                "description": "Cholesterol level",
                "type": "Examination Feature",
                "data_type": "categorical",
                "values": {
                    "1": "Normal",
                    "2": "Above normal", 
                    "3": "Well above normal"
                }
            },
            {
                "name": "gluc",
                "description": "Glucose level", 
                "type": "Examination Feature",
                "data_type": "categorical",
                "values": {
                    "1": "Normal",
                    "2": "Above normal",
                    "3": "Well above normal"
                }
            },
            {
                "name": "smoke",
                "description": "Smoking status",
                "type": "Subjective Feature", 
                "data_type": "binary",
                "values": {
                    "0": "Non-smoker",
                    "1": "Smoker"
                }
            },
            {
                "name": "alco",
                "description": "Alcohol intake",
                "type": "Subjective Feature",
                "data_type": "binary", 
                "values": {
                    "0": "No alcohol consumption",
                    "1": "Alcohol consumption"
                }
            },
            {
                "name": "active",
                "description": "Physical activity",
                "type": "Subjective Feature",
                "data_type": "binary",
                "values": {
                    "0": "Not physically active", 
                    "1": "Physically active"
                }
            }
        ],
        "feature_summary": {
            "total_features": 11,
            "objective_features": 4,
            "examination_features": 4, 
            "subjective_features": 3
        },
        "usage_notes": [
            "Age should be provided in years - it will be automatically converted to days",
            "Gender: use 1 for Male, 0 for Female (note: dataset uses 1=Male, 2=Female but API accepts 0/1)",
            "Blood pressure: ap_hi (systolic) must be greater than ap_lo (diastolic)",
            "Cholesterol and Glucose: 1=Normal, 2=Above Normal, 3=Well Above Normal", 
            "Binary features (smoke, alco, active): 0=No/False, 1=Yes/True"
        ]
    })

@app.route('/', methods=['GET'])
def home():
    """API information endpoint - compatible with Hapi.js format"""
    return jsonify({
        "message": "IllDetect ML Prediction Service",
        "service": "cardiovascular-prediction",
        "version": "1.0.1",
        "status": "active",
        "port": os.environ.get('PORT', 10000),
        "environment": os.environ.get('FLASK_ENV', 'production'),
        "endpoints": {
            "GET /": "API information",
            "GET /api/health": "Health check (compatible with main backend)",
            "GET /api/status": "Service status",
            "POST /api/predict": "Cardiovascular disease prediction",
            "GET /api/model-info": "ML model information",
            "GET /api/dataset-info": "Dataset information",
            "POST /api/convert-age": "Age conversion utility",
            "GET /api/debug": "Debug information"
        },
        "ml_components": {
            "model": "loaded" if model is not None else "not_loaded",
            "scaler": "loaded" if scaler is not None else "not_loaded", 
            "feature_info": "loaded" if feature_info is not None else "not_loaded"
        },
        "integration": {
            "main_backend": "https://backend-api-cgkk.onrender.com",
            "local_backend": "http://localhost:5001",
            "compatible_endpoints": ["health", "predict", "status"],
            "data_format": "standardized for frontend integration"
        }
    })

# Add /api prefix routes for consistency with Hapi.js backend
@app.route('/api/health', methods=['GET'])
def api_health_check():
    """Health check endpoint with /api prefix - compatible with main backend"""
    return health_check()

@app.route('/api/status', methods=['GET'])
def api_status():
    """Service status endpoint"""
    return jsonify({
        "service": "ml-prediction",
        "status": "healthy" if all([model is not None, scaler is not None, feature_info is not None]) else "degraded",
        "timestamp": datetime.datetime.now().isoformat(),
        "uptime": "running",
        "components": {
            "tensorflow_model": "operational" if model is not None else "failed",
            "data_scaler": "operational" if scaler is not None else "failed",
            "feature_processor": "operational" if feature_info is not None else "failed"
        },
        "metrics": {
            "prediction_ready": all([model is not None, scaler is not None, feature_info is not None]),
            "last_model_load": "startup",
            "memory_usage": "normal"
        }
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """ML prediction endpoint with /api prefix"""
    return predict()

@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Model information endpoint with /api prefix"""
    return model_info()

@app.route('/api/dataset-info', methods=['GET'])
def api_dataset_info():
    """Dataset information endpoint with /api prefix"""
    return dataset_info()

@app.route('/api/convert-age', methods=['POST'])
def api_convert_age():
    """Age conversion endpoint with /api prefix"""
    return convert_age()

@app.route('/api/debug', methods=['GET'])
def api_debug_info():
    """Debug information endpoint with /api prefix"""
    return debug_info()

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug information endpoint"""
    try:
        return jsonify({
            "working_directory": os.getcwd(),
            "files_in_directory": os.listdir('.'),
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "feature_info_loaded": feature_info is not None,
            "model_type": str(type(model)) if model is not None else "None",
            "scaler_type": str(type(scaler)) if scaler is not None else "None",
            "feature_info_type": str(type(feature_info)) if feature_info is not None else "None",
            "file_sizes": {
                "my_best_model.h5": os.path.getsize("my_best_model.h5") if os.path.exists("my_best_model.h5") else "Not found",
                "scaler.pkl": os.path.getsize("scaler.pkl") if os.path.exists("scaler.pkl") else "Not found",
                "feature_info.pkl": os.path.getsize("feature_info.pkl") if os.path.exists("feature_info.pkl") else "Not found"
            }
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Try to reload model if not loaded
    if model is None or scaler is None or feature_info is None:
        print("🔄 Health check: Attempting to reload models...")
        load_model_and_preprocessors()
    
    status = {
        "status": "healthy" if all([model is not None, scaler is not None, feature_info is not None]) else "unhealthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_info_loaded": feature_info is not None,
        "working_directory": os.getcwd(),
        "available_files": os.listdir('.') if os.path.exists('.') else []
    }
    
    if all([model is not None, scaler is not None, feature_info is not None]):
        return jsonify(status), 200
    else:
        return jsonify(status), 503

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        # Try to load model
        if not load_model_and_preprocessors():
            return jsonify({
                "error": "Model not loaded and failed to load",
                "status": "error"
            }), 503
    
    try:
        return jsonify({
            "model_type": "Neural Network (Keras)",
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "total_params": int(model.count_params()),
            "layers": len(model.layers),
            "features_count": len(feature_info) if feature_info else "unknown",
            "status": "loaded"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }), 500

@app.route('/convert-age', methods=['POST'])
def convert_age():
    """Convert age between years and days"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No data provided",
                "status": "error"
            }), 400
        
        if 'years' in data:
            years = data['years']
            if not isinstance(years, (int, float)) or years <= 0:
                return jsonify({
                    "error": "Years must be a positive number",
                    "status": "error"
                }), 400
            
            days = convert_age_to_days(years)
            return jsonify({
                "input": f"{years} tahun",
                "output": f"{days} hari",
                "conversion": "years to days",
                "status": "success"
            })
        elif 'days' in data:
            days = data['days']
            if not isinstance(days, (int, float)) or days <= 0:
                return jsonify({
                    "error": "Days must be a positive number",
                    "status": "error"
                }), 400
            
            years = convert_age_to_years(days)
            return jsonify({
                "input": f"{days} hari",
                "output": f"{years} tahun",
                "conversion": "days to years",
                "status": "success"
            })
        else:
            return jsonify({
                "error": "Provide either 'years' or 'days' in request",
                "status": "error"
            }), 400
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }), 500

# Update predict function to handle both direct calls and API calls
@app.route('/predict', methods=['POST'])
def predict():
    """Make cardiovascular disease prediction"""
    try:
        # Validasi model - try to load if not loaded
        if model is None or scaler is None or feature_info is None:
            print("🔄 Predict: Models not loaded, attempting to load...")
            if not load_model_and_preprocessors():
                return jsonify({
                    "success": False,
                    "error": {
                        "message": "ML model tidak tersedia dan gagal dimuat",
                        "type": "model_load_error",
                        "service": "ml-prediction"
                    },
                    "status": "error"
                }), 500
        
        # Ambil data dari request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": {
                    "message": "No prediction data provided",
                    "type": "validation_error",
                    "service": "ml-prediction"
                },
                "status": "error"
            }), 400
        
        # Map frontend field names to backend field names
        field_mapping = {
            'sex': 'gender',
            'systolic': 'ap_hi', 
            'diastolic': 'ap_lo',
            'glucose': 'gluc',
            'smoking': 'smoke',
            'alcohol': 'alco',
            'physical_activity': 'active'
        }
        
        # Convert field names
        converted_data = {}
        for key, value in data.items():
            if key in field_mapping:
                converted_data[field_mapping[key]] = value
            else:
                converted_data[key] = value
        
        # Convert gender from frontend format (0=female, 1=male) to dataset format (1=female, 2=male)
        if 'gender' in converted_data:
            if converted_data['gender'] == 0:  # Frontend: 0 = female
                converted_data['gender'] = 1   # Dataset: 1 = female
            elif converted_data['gender'] == 1:  # Frontend: 1 = male
                converted_data['gender'] = 2     # Dataset: 2 = male
        
        # Validasi input dengan field names yang sudah dikonversi
        required_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                          'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
        for field in required_fields:
            if field not in converted_data:
                original_field = next((k for k, v in field_mapping.items() if v == field), field)
                return jsonify({
                    "success": False,
                    "error": {
                        "message": f"Field '{original_field}' wajib diisi",
                        "type": "validation_error",
                        "service": "ml-prediction"
                    },
                    "status": "error"
                }), 400
        
        # Validasi ranges
        age_years = converted_data['age']
        if not isinstance(age_years, (int, float)) or age_years < 1 or age_years > 120:
            return jsonify({
                "success": False,
                "error": {
                    "message": "Age harus antara 1-120 tahun",
                    "type": "validation_error",
                    "service": "ml-prediction"
                },
                "status": "error"
            }), 400
        
        if converted_data['gender'] not in [1, 2]:
            return jsonify({
                "success": False,
                "error": {
                    "message": "Sex harus 0 (female) atau 1 (male) di frontend",
                    "type": "validation_error",
                    "service": "ml-prediction"
                },
                "status": "error"
            }), 400
        
        if converted_data['height'] <= 0 or converted_data['weight'] <= 0:
            return jsonify({
                "success": False,
                "error": {
                    "message": "Height dan weight harus lebih besar dari 0",
                    "type": "validation_error",
                    "service": "ml-prediction"
                },
                "status": "error"
            }), 400
        
        if converted_data['ap_hi'] <= converted_data['ap_lo']:
            return jsonify({
                "success": False,
                "error": {
                    "message": "Systolic BP harus lebih besar dari Diastolic BP",
                    "type": "validation_error",
                    "service": "ml-prediction"
                },
                "status": "error"
            }), 400
        
        # Validasi categorical fields
        for field in ['cholesterol', 'gluc']:
            if converted_data[field] not in [1, 2, 3]:
                field_name = 'glucose' if field == 'gluc' else field
                return jsonify({
                    "success": False,
                    "error": {
                        "message": f"{field_name} harus 1, 2, atau 3",
                        "type": "validation_error",
                        "service": "ml-prediction"
                    },
                    "status": "error"
                }), 400
        
        for field, original_name in [('smoke', 'smoking'), ('alco', 'alcohol'), ('active', 'physical_activity')]:
            if converted_data[field] not in [0, 1]:
                return jsonify({
                    "success": False,
                    "error": {
                        "message": f"{original_name} harus 0 atau 1",
                        "type": "validation_error",
                        "service": "ml-prediction"
                    },
                    "status": "error"
                }), 400
        
        # Convert age to days
        age_days = convert_age_to_days(converted_data['age'])
        
        # Prepare input data
        input_data = np.array([[
            age_days,
            converted_data['gender'],
            converted_data['height'],
            converted_data['weight'],
            converted_data['ap_hi'],
            converted_data['ap_lo'],
            converted_data['cholesterol'],
            converted_data['gluc'],
            converted_data['smoke'],
            converted_data['alco'],
            converted_data['active']
        ]])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_prob = model.predict(input_scaled)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)
        
        # Calculate BMI
        bmi = round(converted_data['weight'] / ((converted_data['height']/100) ** 2), 2)
        
        # Format response to match main backend format
        response = {
            "success": True,
            "data": {
                "prediction": int(prediction),
                "confidence": round(float(confidence), 4),
                "probability": round(float(prediction_prob), 4),
                "risk_level": "HIGH" if prediction == 1 else "LOW",
                "result_message": "RISIKO TINGGI - Disarankan konsultasi dengan dokter" if prediction == 1 else "RISIKO RENDAH - Pertahankan gaya hidup sehat",
                "interpretation": "Berdasarkan data yang diberikan, terdapat indikasi risiko penyakit kardiovaskular" if prediction == 1 else "Berdasarkan data yang diberikan, risiko penyakit kardiovaskular relatif rendah",
                "patient_data": {
                    "age_years": converted_data['age'],
                    "age_days": age_days,
                    "gender": "Female" if converted_data['gender'] == 1 else "Male",  # Updated mapping
                    "height": converted_data['height'],
                    "weight": converted_data['weight'],
                    "bmi": bmi,
                    "bmi_category": "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese",
                    "blood_pressure": f"{converted_data['ap_hi']}/{converted_data['ap_lo']}",
                    "cholesterol_level": ["Normal", "Above Normal", "Well Above Normal"][converted_data['cholesterol']-1],
                    "glucose_level": ["Normal", "Above Normal", "Well Above Normal"][converted_data['gluc']-1],
                    "lifestyle": {
                        "smoking": "Yes" if converted_data['smoke'] == 1 else "No",
                        "alcohol": "Yes" if converted_data['alco'] == 1 else "No",
                        "physical_activity": "Yes" if converted_data['active'] == 1 else "No"
                    }
                }
            },
            "metadata": {
                "service": "ml-prediction",
                "model_version": "1.0",
                "prediction_id": f"pred_{int(datetime.datetime.now().timestamp())}",
                "timestamp": datetime.datetime.now().isoformat(),
                "processing_time": "< 1s"
            },
            "status": "success"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": {
                "message": str(e),
                "type": "prediction_error",
                "service": "ml-prediction",
                "traceback": traceback.format_exc()
            },
            "status": "error"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "status": "error"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The request method is not allowed for this endpoint",
        "status": "error"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status": "error"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🤖 Starting IllDetect ML Service on port {port}")
    print(f"🔗 Main backend integration:")
    print(f"   Production: https://backend-api-cgkk.onrender.com")
    print(f"   Local: http://localhost:5001")
    print(f"🎯 ML endpoints available with /api prefix for consistency")
    print(f"🌐 CORS enabled for backend integration")
    print(f"📡 Health check available at /ping")
    app.run(debug=False, host='0.0.0.0', port=port)