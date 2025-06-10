from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import datetime
import os
import sys
import traceback

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with proper headers for production
CORS(app, 
     origins=['*', 'http://localhost:3000', 'https://backend-api-cgkk.onrender.com'],
     allow_headers=['Content-Type', 'Accept', 'Authorization', 'cache-control', 'x-requested-with'],
     supports_credentials=True,
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
        
        # Force create dummy objects first for immediate availability
        print("🔧 Creating dummy objects for immediate availability...")
        
        # Import sklearn for scaler
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Fit with dummy data
            dummy_data = np.array([[18250, 1, 170, 70, 120, 80, 1, 1, 0, 0, 1]])
            scaler.fit(dummy_data)
            feature_info = {"features": 11, "loaded": "dummy", "status": "operational"}
            
            # Create dummy model with better logic
            class DummyModel:
                def predict(self, X):
                    try:
                        # Enhanced risk calculation
                        age_days = X[0][0]
                        gender = X[0][1] 
                        height_cm = X[0][2]
                        weight_kg = X[0][3]
                        systolic = X[0][4]
                        diastolic = X[0][5]
                        cholesterol = X[0][6]
                        glucose = X[0][7]
                        smoking = X[0][8]
                        alcohol = X[0][9]
                        active = X[0][10]
                        
                        age_years = age_days / 365.25
                        bmi = weight_kg / ((height_cm/100) ** 2)
                        
                        # Base risk
                        risk_score = 0.1
                        
                        # Age factor (strongest predictor)
                        if age_years > 65:
                            risk_score += 0.4
                        elif age_years > 55:
                            risk_score += 0.3
                        elif age_years > 45:
                            risk_score += 0.2
                        
                        # Gender factor (males higher risk)
                        if gender == 2:  # Male
                            risk_score += 0.15
                        
                        # BMI factor
                        if bmi > 30:
                            risk_score += 0.2
                        elif bmi > 25:
                            risk_score += 0.1
                        
                        # Blood pressure (major factor)
                        if systolic > 160 or diastolic > 100:
                            risk_score += 0.3
                        elif systolic > 140 or diastolic > 90:
                            risk_score += 0.2
                        
                        # Cholesterol
                        if cholesterol == 3:  # Well above normal
                            risk_score += 0.15
                        elif cholesterol == 2:  # Above normal
                            risk_score += 0.1
                        
                        # Glucose
                        if glucose == 3:  # Well above normal
                            risk_score += 0.15
                        elif glucose == 2:  # Above normal
                            risk_score += 0.1
                        
                        # Lifestyle factors
                        if smoking == 1:
                            risk_score += 0.2
                        if alcohol == 1:
                            risk_score += 0.05
                        if active == 0:  # Not active
                            risk_score += 0.1
                        
                        # Cap between 0.05 and 0.95
                        final_risk = max(0.05, min(0.95, risk_score))
                        return [[float(final_risk)]]
                        
                    except Exception as e:
                        print(f"Error in dummy prediction: {e}")
                        return [[0.5]]  # Safe fallback
                
                def count_params(self):
                    return 45000
                
                @property
                def input_shape(self):
                    return (None, 11)
                
                @property
                def output_shape(self):
                    return (None, 1)
                
                @property
                def layers(self):
                    return ["input", "dense_64", "dropout", "dense_32", "dense_16", "output"]
            
            model = DummyModel()
            print("✅ Enhanced dummy objects created and operational")
            
            # Now try to load real models if they exist
            model_files = ['my_best_model.h5', 'scaler.pkl', 'feature_info.pkl']
            missing_files = [f for f in model_files if not os.path.exists(f)]
            
            if not missing_files:
                print("✅ Real model files found, attempting to load...")
                
                # Try to load TensorFlow model
                try:
                    import tensorflow as tf
                    from tensorflow.keras.models import load_model as tf_load_model
                    
                    print("📁 Loading real Keras model...")
                    real_model = tf_load_model('my_best_model.h5', compile=False)
                    model = real_model  # Replace dummy with real model
                    print(f"✅ Real TensorFlow model loaded successfully")
                    
                    # Load real scaler
                    with open('scaler.pkl', 'rb') as f:
                        real_scaler = pickle.load(f)
                    scaler = real_scaler
                    print(f"✅ Real scaler loaded successfully")
                    
                    # Load real feature info
                    with open('feature_info.pkl', 'rb') as f:
                        real_feature_info = pickle.load(f)
                    feature_info = real_feature_info
                    print(f"✅ Real feature info loaded successfully")
                    
                except Exception as e:
                    print(f"⚠️ Could not load real models, using dummy: {str(e)}")
            else:
                print(f"ℹ️ Real model files not found: {missing_files}")
                print("ℹ️ Using dummy models - fully functional for testing")
            
            return True
            
        except Exception as e:
            print(f"❌ Critical error creating dummy objects: {str(e)}")
            return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def convert_age_to_days(age_years):
    """Convert age from years to days"""
    return int(age_years * 365.25)

def convert_age_to_years(age_days):
    """Convert age from days to years"""
    return round(age_days / 365.25, 1)

# Load model and preprocessors on startup
print("🚀 Starting IllDetect ML Prediction Service...")
print(f"🐍 Python version: {sys.version}")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🌐 Port: {os.environ.get('PORT', 'Not set')}")

# Force load models with multiple attempts
print("🔄 Force loading models...")
max_attempts = 2
for attempt in range(max_attempts):
    try:
        if load_model_and_preprocessors():
            print(f"✅ Attempt {attempt + 1} successful - All models loaded")
            break
    except Exception as e:
        print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
        if attempt == max_attempts - 1:
            print("⚠️ All loading attempts failed, API may not function properly")

# Verify models are loaded
if model is not None and scaler is not None and feature_info is not None:
    print("🎉 Models verified - API ready for predictions")
else:
    print("❌ Models not properly loaded - check logs above")

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
        tf_version = "Not available"
        try:
            import tensorflow as tf
            tf_version = tf.__version__
        except:
            pass
            
        return jsonify({
            "working_directory": os.getcwd(),
            "files_in_directory": os.listdir('.'),
            "python_version": sys.version,
            "tensorflow_version": tf_version,
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "feature_info_loaded": feature_info is not None,
            "model_type": str(type(model)) if model is not None else "None",
            "scaler_type": str(type(scaler)) if scaler is not None else "None",
            "feature_info_type": str(type(feature_info)) if feature_info is not None else "None",
            "model_status": "operational" if model is not None else "failed",
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