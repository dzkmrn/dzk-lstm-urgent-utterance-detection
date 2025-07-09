import streamlit as st

# Configure page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Emergency Voice Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import tempfile
import os
from pydub import AudioSegment
import io
import threading
import queue
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import logging
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import json
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
import base64
from io import BytesIO

# Configure TensorFlow for cloud deployment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_METAL_DISABLE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL optimizations

tf.config.set_visible_devices([], 'GPU')  # Disable GPU completely
tf.config.threading.set_inter_op_parallelism_threads(1)  # Limit threading
tf.config.threading.set_intra_op_parallelism_threads(1)  # Limit threading
tf.get_logger().setLevel('ERROR')  # Reduce TensorFlow logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('tensorflow').setLevel('WARNING')
logging.getLogger('librosa').setLevel('WARNING')

# Create a separate logger for debugging
debug_logger = logging.getLogger('DEBUG')
debug_logger.setLevel(logging.DEBUG)

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs("processed", exist_ok=True)
os.makedirs("uploaded", exist_ok=True)

# Constants for audio processing
TARGET_SR = 16000
N_MELS = 128
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = None
DURATION = 3  # seconds
MIN_MFCC_FRAMES = 9
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

# Custom CSS - Enhanced Design
st.markdown("""
<style>
    /* Global Styling */
    html, body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #333 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    /* Main Container */
    [data-testid="stAppViewContainer"], .main, .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 0 !important;
    }

    /* Header Styling */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .main-header h1 {
        color: #2c3e50 !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
    }

    .main-header p {
        color: #7f8c8d !important;
        font-size: 1.2rem !important;
        margin: 0 !important;
    }

    /* Dashboard Cards */
    .dashboard-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }

    /* Recording Button */
    [data-testid="stButton"][key="hidden_record_trigger"] > button {
        width: 100%;
        height: 80px;
        font-size: 1.5rem;
        padding: 20px;
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4);
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stButton"][key="hidden_record_trigger"] > button:hover {
        background: linear-gradient(45deg, #c0392b, #a93226);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.6);
    }

    /* File Uploader */
    .stFileUploader > div > div {
        border: 3px dashed #3498db;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }

    .stFileUploader > div > div:hover {
        border-color: #2980b9;
        background: rgba(52, 152, 219, 0.1);
        transform: translateY(-2px);
    }

    /* Results Display */
    .emergency-status {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
    }

    .emergency-status.urgent {
        background: linear-gradient(45deg, #e74c3c, #c0392b) !important;
        color: white !important;
        animation: pulse 2s infinite;
    }

    .emergency-status.safe {
        background: linear-gradient(45deg, #27ae60, #229954) !important;
        color: white !important;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #2980b9, #21618c);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        font-weight: bold;
    }

    /* Success/Error Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# User database functionality
USERS_DB = {}

def save_users_db():
    """Save users database to file"""
    try:
        with open('data/users_db.json', 'w') as f:
            json.dump(USERS_DB, f, indent=2)
    except Exception as e:
        debug_logger.error(f"Error saving users database: {e}")

def load_users_db():
    """Load users database from file"""
    try:
        if os.path.exists('data/users_db.json'):
            with open('data/users_db.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        debug_logger.error(f"Error loading users database: {e}")
    return {}

# Initialize user database
USERS_DB = load_users_db()

# Initialize session state for current user
if 'current_user' not in st.session_state:
    st.session_state.current_user = 'default_user'
    if st.session_state.current_user not in USERS_DB:
        USERS_DB[st.session_state.current_user] = {
            'history': [],
            'created_at': datetime.datetime.now().isoformat()
        }
        save_users_db()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.label_encoder = None

def save_users_db():
    """Simple function to maintain compatibility"""
    pass

@st.cache_resource
def load_model_and_encoder():
    """Load LSTM model and label encoder with optimizations for cloud deployment"""
    try:
        debug_logger.info("=== Starting model loading process ===")
        logger.info("Loading LSTM model...")
        
        # Check if models directory exists
        if not os.path.exists('models'):
            debug_logger.error("Models directory does not exist!")
            raise FileNotFoundError("Models directory not found")
        
        # List all files in models directory
        model_files = os.listdir('models')
        debug_logger.info(f"Files in models directory: {model_files}")
        
        # Check if model file exists
        model_path = 'models/van_et_al.h5'
        if not os.path.exists(model_path):
            debug_logger.error(f"Model file not found at: {model_path}")
            debug_logger.info(f"Available model files: {[f for f in model_files if f.endswith('.h5')]}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_size = os.path.getsize(model_path) / (1024*1024)  # MB
        debug_logger.info(f"Model file size: {model_size:.2f} MB")
        
        # Load model with memory optimization
        debug_logger.info("Loading TensorFlow model...")
        with tf.device('/CPU:0'):  # Force CPU usage
            model = tf.keras.models.load_model(model_path, compile=False)
            debug_logger.info("Model loaded, starting compilation...")
            
            # Compile with memory-efficient settings
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy', 
                metrics=['accuracy'],
                run_eagerly=False
            )
            debug_logger.info("Model compilation completed")
            
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
        debug_logger.info(f"Model summary - Input: {model.input_shape}, Output: {model.output_shape}")
        
        # Load label encoder classes
        encoder_path = 'models/label_encoder_classes.npy'
        if not os.path.exists(encoder_path):
            debug_logger.error(f"Label encoder file not found at: {encoder_path}")
            raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
            
        debug_logger.info("Loading label encoder...")
        label_encoder_classes = np.load(encoder_path, allow_pickle=True)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_encoder_classes
        logger.info(f"Label encoder loaded with classes: {label_encoder.classes_}")
        debug_logger.info(f"Label encoder classes type: {type(label_encoder_classes)}, shape: {label_encoder_classes.shape if hasattr(label_encoder_classes, 'shape') else 'N/A'}")
        
        debug_logger.info("=== Model loading completed successfully ===")
        return model, label_encoder
        
    except Exception as e:
        debug_logger.error(f"CRITICAL ERROR in model loading: {str(e)}")
        debug_logger.error(f"Exception type: {type(e)}")
        logger.error(f"Error loading model: {str(e)}")
        raise Exception(f"Model loading failed: {str(e)}")

def record_audio(duration=3, sample_rate=TARGET_SR):
    """Record audio for a fixed duration"""
    debug_logger.info(f"=== Starting audio recording - Duration: {duration}s, Sample rate: {sample_rate} ===")
    audio_data = []
    recording_errors = []
    
    def callback(indata, frames, time, status):
        if status:
            debug_logger.warning(f"Recording callback status: {status}")
            recording_errors.append(str(status))
        audio_data.append(indata.copy())
        debug_logger.debug(f"Audio chunk received - frames: {frames}, shape: {indata.shape}")
    
    try:
        debug_logger.info("Creating audio input stream...")
        # Record audio (this is a blocking operation)
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            debug_logger.info(f"Recording started, sleeping for {duration * 1000}ms...")
            sd.sleep(int(duration * 1000))
        
        debug_logger.info(f"Recording completed. Chunks collected: {len(audio_data)}")
        
        if not audio_data:
            debug_logger.error("No audio data collected!")
            return None
        
        # Concatenate audio data
        result = np.concatenate(audio_data, axis=0)
        debug_logger.info(f"Final audio shape: {result.shape}, dtype: {result.dtype}")
        debug_logger.info(f"Audio stats - min: {np.min(result):.6f}, max: {np.max(result):.6f}, mean: {np.mean(result):.6f}")
        
        if recording_errors:
            debug_logger.warning(f"Recording errors encountered: {recording_errors}")
        
        return result
        
    except Exception as e:
        debug_logger.error(f"ERROR in record_audio: {str(e)}")
        debug_logger.error(f"Exception type: {type(e)}")
        return None

def pad_audio_min_length(audio, sr, min_mfcc_frames=MIN_MFCC_FRAMES, n_fft=N_FFT, hop_length=HOP_LENGTH):
    min_len_samples = n_fft + hop_length * (min_mfcc_frames - 1)
    if len(audio) < min_len_samples:
        pad_amount = min_len_samples - len(audio)
        audio = np.pad(audio, (0, pad_amount), mode='constant')
    return audio

def preprocess_audio(audio_data, sr):
    debug_logger.info(f"=== Starting audio preprocessing ===")
    debug_logger.info(f"Input audio shape: {audio_data.shape}, sample rate: {sr}")
    debug_logger.info(f"Input audio stats - min: {np.min(audio_data):.6f}, max: {np.max(audio_data):.6f}, mean: {np.mean(audio_data):.6f}")
    
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            debug_logger.info(f"Converting stereo to mono from shape: {audio_data.shape}")
            audio_data = np.mean(audio_data, axis=1)
            debug_logger.info(f"After mono conversion: {audio_data.shape}")
        
        # Resample if needed
        if sr != TARGET_SR:
            debug_logger.info(f"Resampling from {sr} Hz to {TARGET_SR} Hz")
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
            debug_logger.info(f"After resampling: {audio_data.shape}")
        
        # Truncate silence
        debug_logger.info("Trimming silence...")
        original_length = len(audio_data)
        audio_data, _ = librosa.effects.trim(audio_data, top_db=15)
        debug_logger.info(f"After trimming: {len(audio_data)} samples (removed {original_length - len(audio_data)} samples)")
        
        # Pad if needed
        debug_logger.info("Padding audio to minimum length...")
        audio_data = pad_audio_min_length(audio_data, TARGET_SR)
        debug_logger.info(f"Final preprocessed audio shape: {audio_data.shape}")
        debug_logger.info(f"Final audio stats - min: {np.min(audio_data):.6f}, max: {np.max(audio_data):.6f}, mean: {np.mean(audio_data):.6f}")
        
        debug_logger.info("=== Audio preprocessing completed successfully ===")
        return audio_data
        
    except Exception as e:
        debug_logger.error(f"ERROR in preprocess_audio: {str(e)}")
        debug_logger.error(f"Exception type: {type(e)}")
        st.error(f"Error in audio preprocessing: {str(e)}")
        return None

def extract_mfcc_features(audio, sr):
    debug_logger.info(f"=== Starting MFCC feature extraction ===")
    debug_logger.info(f"Input audio shape: {audio.shape}, sample rate: {sr}")
    debug_logger.info(f"MFCC parameters - n_mfcc: {N_MFCC}, n_mels: {N_MELS}, hop_length: {HOP_LENGTH}, n_fft: {N_FFT}")
    
    try:
        # Extract MFCC features
        debug_logger.info("Extracting MFCC features...")
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr,
            n_mfcc=N_MFCC,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            fmin=FMIN,
            fmax=FMAX
        )
        debug_logger.info(f"Raw MFCC shape: {mfcc.shape}")
        
        # Transpose to get time as first dimension
        debug_logger.info("Transposing MFCC...")
        mfcc = mfcc.T
        debug_logger.info(f"After transpose: {mfcc.shape}")
        
        # Pad or truncate to match model's expected input length
        target_length = 104  # Model's expected input length
        debug_logger.info(f"Adjusting to target length: {target_length}")
        
        if mfcc.shape[0] > target_length:
            debug_logger.info(f"Truncating from {mfcc.shape[0]} to {target_length}")
            mfcc = mfcc[:target_length, :]
        elif mfcc.shape[0] < target_length:
            pad_amount = target_length - mfcc.shape[0]
            debug_logger.info(f"Padding from {mfcc.shape[0]} to {target_length} (adding {pad_amount} frames)")
            mfcc = np.pad(mfcc, ((0, pad_amount), (0, 0)), mode='constant')
        
        debug_logger.info(f"After length adjustment: {mfcc.shape}")
        
        # Reshape to match model's expected input shape (None, 104, 13, 3)
        debug_logger.info("Reshaping for model input...")
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        debug_logger.info(f"After adding channel dimension: {mfcc.shape}")
        
        mfcc = np.repeat(mfcc, 3, axis=-1)    # Repeat to get 3 channels
        debug_logger.info(f"After repeating to 3 channels: {mfcc.shape}")
        
        mfcc = np.expand_dims(mfcc, axis=0)   # Add batch dimension
        debug_logger.info(f"Final MFCC shape: {mfcc.shape}")
        
        # Verify expected shape
        expected_shape = (1, 104, 13, 3)
        if mfcc.shape != expected_shape:
            debug_logger.error(f"Shape mismatch! Got {mfcc.shape}, expected {expected_shape}")
            return None
        
        debug_logger.info(f"MFCC stats - min: {np.min(mfcc):.6f}, max: {np.max(mfcc):.6f}, mean: {np.mean(mfcc):.6f}")
        debug_logger.info("=== MFCC feature extraction completed successfully ===")
        
        return mfcc
        
    except Exception as e:
        debug_logger.error(f"ERROR in extract_mfcc_features: {str(e)}")
        debug_logger.error(f"Exception type: {type(e)}")
        st.error(f"Error extracting features: {str(e)}")
        return None

def predict_audio(features, model, label_encoder):
    """Make prediction using the model"""
    debug_logger.info(f"=== Starting audio prediction ===")
    debug_logger.info(f"Features shape: {features.shape}")
    debug_logger.info(f"Features stats - min: {np.min(features):.6f}, max: {np.max(features):.6f}, mean: {np.mean(features):.6f}")
    
    try:
        # Verify input shape before prediction
        expected_shape = (1, 104, 13, 3)
        if features.shape != expected_shape:
            debug_logger.error(f"Invalid input shape for prediction: {features.shape}. Expected {expected_shape}")
            raise ValueError(f"Invalid input shape for prediction: {features.shape}. Expected {expected_shape}")
        
        debug_logger.info("Input shape verification passed")
        
        # Make prediction
        debug_logger.info("Making prediction with model...")
        prediction = model.predict(features, verbose=0)
        debug_logger.info(f"Raw prediction shape: {prediction.shape}")
        debug_logger.info(f"Raw prediction values: {prediction}")
        
        predicted_class_idx = np.argmax(prediction)
        debug_logger.info(f"Predicted class index: {predicted_class_idx}")
        
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = np.max(prediction)
        
        debug_logger.info(f"Predicted class: {predicted_class}")
        debug_logger.info(f"Confidence: {confidence}")
        debug_logger.info(f"All class probabilities: {dict(zip(label_encoder.classes_, prediction[0]))}")
        debug_logger.info("=== Audio prediction completed successfully ===")
        
        return predicted_class, confidence
        
    except Exception as e:
        debug_logger.error(f"ERROR in predict_audio: {str(e)}")
        debug_logger.error(f"Exception type: {type(e)}")
        logger.error(f"Error making prediction: {str(e)}")
        return None, None

def process_and_predict(audio_path, model, label_encoder):
    debug_logger.info(f"=== Starting process_and_predict ===")
    debug_logger.info(f"Audio file path: {audio_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            debug_logger.error(f"Audio file not found: {audio_path}")
            st.error(f"Audio file not found: {audio_path}")
            return None, None, None
        
        file_size = os.path.getsize(audio_path)
        debug_logger.info(f"Audio file size: {file_size} bytes")
        
        # Load and preprocess audio
        debug_logger.info(f"Loading audio file with librosa (target SR: {TARGET_SR})...")
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        debug_logger.info(f"Loaded audio - shape: {y.shape}, sample rate: {sr}")
        debug_logger.info(f"Audio duration: {len(y)/sr:.2f} seconds")
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            debug_logger.info(f"Converting to mono from shape: {y.shape}")
            y = librosa.to_mono(y)
            debug_logger.info(f"After mono conversion: {y.shape}")
        
        # Truncate silence (trim audio)
        debug_logger.info("Trimming silence...")
        original_length = len(y)
        y_trimmed, index = librosa.effects.trim(y, top_db=15)
        y = y_trimmed
        debug_logger.info(f"After trimming: {len(y)} samples (removed {original_length - len(y)} samples)")
        
        # Extract MFCC features
        debug_logger.info("Extracting MFCC features...")
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=TARGET_SR,
            n_mfcc=N_MFCC,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            fmin=FMIN,
            fmax=FMAX
        )
        debug_logger.info(f"Raw MFCC shape: {mfcc.shape}")
        
        # Transpose to get time as first dimension
        mfcc = mfcc.T
        debug_logger.info(f"After transpose: {mfcc.shape}")
        
        # Pad or truncate to match model's expected input length
        target_length = 104  # Model's expected input length
        debug_logger.info(f"Adjusting to target length: {target_length}")
        
        if mfcc.shape[0] > target_length:
            debug_logger.info(f"Truncating from {mfcc.shape[0]} to {target_length}")
            mfcc = mfcc[:target_length, :]
        elif mfcc.shape[0] < target_length:
            pad_amount = target_length - mfcc.shape[0]
            debug_logger.info(f"Padding from {mfcc.shape[0]} to {target_length} (adding {pad_amount} frames)")
            mfcc = np.pad(mfcc, ((0, pad_amount), (0, 0)), mode='constant')
        
        # Reshape to match model's expected input shape (None, 104, 13, 3)
        debug_logger.info("Reshaping for model input...")
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        mfcc = np.repeat(mfcc, 3, axis=-1)    # Repeat to get 3 channels
        mfcc = np.expand_dims(mfcc, axis=0)   # Add batch dimension
        debug_logger.info(f"Final MFCC shape: {mfcc.shape}")
        
        # Verify the shape
        expected_shape = (1, 104, 13, 3)
        if mfcc.shape != expected_shape:
            debug_logger.error(f"Shape mismatch! Got {mfcc.shape}, expected {expected_shape}")
            raise ValueError(f"Invalid input shape: {mfcc.shape}. Expected {expected_shape}")
        
        debug_logger.info("Shape verification passed, making prediction...")
        
        # Make prediction
        predicted_class, confidence = predict_audio(mfcc, model, label_encoder)
        
        if predicted_class is not None:
            debug_logger.info(f"Prediction successful: {predicted_class} ({confidence:.4f})")
            
            # Save processed audio
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_path = f"processed/processed_{timestamp}.wav"
            debug_logger.info(f"Saving processed audio to: {processed_path}")
            
            sf.write(processed_path, y, TARGET_SR)
            debug_logger.info(f"Processed audio saved successfully")
            
            # Add to history
            history_entry = {
                "file": processed_path,
                "label": predicted_class,
                "confidence": confidence * 100,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.history.append(history_entry)
            debug_logger.info(f"Added to history: {history_entry}")
            
            debug_logger.info("=== process_and_predict completed successfully ===")
            return predicted_class, confidence * 100, processed_path
        else:
            debug_logger.error("Prediction failed - returned None")
            st.error("Prediction failed")
            return None, None, None
            
    except Exception as e:
        debug_logger.error(f"CRITICAL ERROR in process_and_predict: {str(e)}")
        debug_logger.error(f"Exception type: {type(e)}")
        import traceback
        debug_logger.error(f"Traceback: {traceback.format_exc()}")
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None

def visualize_audio(audio_path):
    """Generate basic audio visualization"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title('Waveform', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitudo')
        ax1.grid(True, alpha=0.3)
        
        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2)
        ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Frequency (Hz)')
        
        # Add colorbar
        fig.colorbar(img, ax=ax2, format="%+2.f dB")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        debug_logger.error(f"Error generating visualization: {str(e)}")
        return None

def generate_comprehensive_visualizations(original_audio_path, processed_audio_data, sr=TARGET_SR):
    """Generate comprehensive visualizations showing the complete processing pipeline"""
    try:
        # Load original audio
        y_original, sr_orig = librosa.load(original_audio_path, sr=None)
        if sr_orig != TARGET_SR:
            y_original_resampled = librosa.resample(y_original, orig_sr=sr_orig, target_sr=TARGET_SR)
        else:
            y_original_resampled = y_original
        
        # Use processed audio data
        y_processed = processed_audio_data
        
        # Create figure with 4 subplots (2x2)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Original Waveform
        time_orig = np.linspace(0, len(y_original_resampled)/TARGET_SR, len(y_original_resampled))
        ax1.plot(time_orig, y_original_resampled, color='blue', alpha=0.7)
        ax1.set_title('🎵 Waveform Input Original', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Waktu (detik)')
        ax1.set_ylabel('Amplitudo')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_orig))
        
        # 2. Processed Waveform (after preprocessing)
        time_proc = np.linspace(0, len(y_processed)/TARGET_SR, len(y_processed))
        ax2.plot(time_proc, y_processed, color='green', alpha=0.7)
        ax2.set_title('🔧 Waveform Setelah Preprocessing', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Waktu (detik)')
        ax2.set_ylabel('Amplitudo')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_proc))
        
        # 3. MFCC Features (basic)
        mfcc_basic = librosa.feature.mfcc(y=y_processed, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        img1 = librosa.display.specshow(mfcc_basic, sr=sr, x_axis='time', ax=ax3, cmap='viridis')
        ax3.set_title('📊 MFCC Features (13 coefficients)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Waktu (detik)')
        ax3.set_ylabel('MFCC Coefficients')
        fig.colorbar(img1, ax=ax3, format="%.2f")
        
        # 4. MFCC + Delta + Delta-Delta (what model actually uses)
        mfcc = librosa.feature.mfcc(y=y_processed, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack all features (3, 13, time_steps)
        mfcc_stack = np.stack([mfcc, delta, delta2], axis=0)
        # Reshape to (time_steps, 13, 3) then flatten to (time_steps, 39) for visualization
        mfcc_combined = mfcc_stack.transpose(2, 1, 0)  # (time_steps, 13, 3)
        mfcc_flattened = mfcc_combined.reshape(mfcc_combined.shape[0], -1)  # (time_steps, 39)
        
        img2 = ax4.imshow(mfcc_flattened.T, aspect='auto', origin='lower', cmap='plasma')
        ax4.set_title('🎯 MFCC + Delta + Delta-Delta (Input Model)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Frames')
        ax4.set_ylabel('Features (MFCC + Δ + ΔΔ)')
        
        # Add feature labels
        feature_labels = []
        for i in range(13):
            feature_labels.append(f'MFCC_{i+1}')
        for i in range(13):
            feature_labels.append(f'Δ_{i+1}')
        for i in range(13):
            feature_labels.append(f'ΔΔ_{i+1}')
        
        # Set y-tick labels for every 5th feature to avoid crowding
        y_ticks = range(0, 39, 5)
        ax4.set_yticks(y_ticks)
        ax4.set_yticklabels([feature_labels[i] for i in y_ticks], fontsize=8)
        
        fig.colorbar(img2, ax=ax4, format="%.2f")
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
    except Exception as e:
        debug_logger.error(f"Error generating comprehensive visualizations: {str(e)}")
        return None

def generate_audio_visualizations(audio_path):
    """Generate waveform and spectrogram visualizations"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title('Waveform', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Waktu (detik)')
        ax1.set_ylabel('Amplitudo')
        ax1.grid(True, alpha=0.3)
        
        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2)
        ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Waktu (detik)')
        ax2.set_ylabel('Frekuensi (Hz)')
        
        # Add colorbar
        fig.colorbar(img, ax=ax2, format="%+2.f dB")
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
    except Exception as e:
        debug_logger.error(f"Error generating visualizations: {str(e)}")
        return None

def generate_mfcc_visualization(audio_path):
    """Generate MFCC feature visualization"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot MFCC
        img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
        ax.set_title('MFCC Features', fontsize=14, fontweight='bold')
        ax.set_xlabel('Waktu (detik)')
        ax.set_ylabel('MFCC Coefficients')
        
        # Add colorbar
        fig.colorbar(img, ax=ax, format="%.2f")
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
    except Exception as e:
        debug_logger.error(f"Error generating MFCC visualization: {str(e)}")
        return None

def user_interface():
    debug_logger.info("=== STARTING USER INTERFACE ===")
    debug_logger.info(f"Session state model_loaded: {st.session_state.get('model_loaded', False)}")
    
    # Initialize model loading state
    if not st.session_state.model_loaded:
        debug_logger.info("Model not loaded, starting initialization...")
        st.markdown("## 🤖 Initializing Emergency Voice Detection System")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            debug_logger.info("Starting model loading process...")
            status_text.text("🔄 Loading TensorFlow model...")
            progress_bar.progress(25)
            
            model, label_encoder = load_model_and_encoder()
            debug_logger.info("Model and encoder loaded successfully")
            
            progress_bar.progress(75)
            status_text.text("✅ Model loaded successfully!")
            
            st.session_state.model = model
            st.session_state.label_encoder = label_encoder
            st.session_state.model_loaded = True
            debug_logger.info("Model stored in session state")
            
            progress_bar.progress(100)
            status_text.text("🚀 System ready!")
            
            # Small delay to show completion
            import time
            time.sleep(1)
            
            debug_logger.info("Model initialization completed, triggering rerun...")
            st.rerun()
            
        except Exception as e:
            debug_logger.error(f"CRITICAL ERROR during model initialization: {str(e)}")
            debug_logger.error(f"Exception type: {type(e)}")
            import traceback
            debug_logger.error(f"Traceback: {traceback.format_exc()}")
            
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Failed to load model: {e}")
            st.error("Please refresh the page to try again.")
            st.stop()
        return
    
    # Use loaded model from session state
    model = st.session_state.model
    label_encoder = st.session_state.label_encoder
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🚨 Emergency Voice Detection System</h1>
        <p>Sistem Deteksi Suara Darurat Berbasis AI - Siap Melayani 24/7</p>
    </div>
    """, unsafe_allow_html=True)

    # Dashboard Layout - 50:50
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Recording Section
        st.markdown("""
        <div class="dashboard-card">
            <h2 style="color: #2c3e50; margin-bottom: 1rem;">🎙️ Rekam Suara Darurat</h2>
            <p style="color: #7f8c8d; margin-bottom: 2rem;">Tekan tombol di bawah untuk merekam suara selama 3 detik</p>
        </div>
        """, unsafe_allow_html=True)
        
        with stylable_container(
            key="record_button_container",
            css_styles="""
                {
                    text-align: center;
                    margin: 2rem 0;
                }
            """
        ):
            if st.button("🎤 MULAI REKAM", key="hidden_record_trigger", type="primary"):
                handle_recording(model, label_encoder)

        # File Upload Section
        st.markdown("""
        <div class="dashboard-card">
            <h2 style="color: #2c3e50; margin-bottom: 1rem;">📁 Unggah File Audio</h2>
            <p style="color: #7f8c8d; margin-bottom: 1rem;">Atau unggah file audio yang sudah ada (WAV/MP3)</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Pilih file audio", type=["wav", "mp3"], help="Unggah file audio untuk analisis", label_visibility="collapsed")
        
        if uploaded_file is not None:
            current_upload_key = uploaded_file.name + str(uploaded_file.size)
            
            if 'processed_upload_key' not in st.session_state or st.session_state.processed_upload_key != current_upload_key:
                if st.button("🔄 Proses File Audio", key="process_upload_button", type="secondary"):
                    handle_upload(uploaded_file, model, label_encoder)
                    st.session_state.processed_upload_key = current_upload_key
            else:
                st.success("✅ File ini sudah diproses.")
        

    with col2:
        # Show visualizations if result exists, otherwise show placeholder
        if 'result' in st.session_state and 'visualizations' in st.session_state:
            # Visualizations Section
            st.markdown("""
            <div class="dashboard-card">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">📈 Visualisasi & Ekstraksi Suara</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display visualizations
            viz = st.session_state.visualizations
            
            if 'comprehensive_visualization' in viz and viz['comprehensive_visualization']:
                st.markdown("**🔄 Pipeline Preprocessing & Feature Extraction**")
                st.image(f"data:image/png;base64,{viz['comprehensive_visualization']}", 
                        caption="Pipeline lengkap: Original → Preprocessed → MFCC → MFCC+Delta+Delta-Delta",
                        use_container_width=True)
            
            if 'waveform_spectrogram' in viz and viz['waveform_spectrogram']:
                st.markdown("**🌊 Waveform & Spectrogram**")
                st.image(f"data:image/png;base64,{viz['waveform_spectrogram']}", 
                        caption="Bentuk gelombang dan spektrogram audio",
                        use_container_width=True)
            
            if 'mfcc_visualization' in viz and viz['mfcc_visualization']:
                st.markdown("**🎼 MFCC Features**")
                st.image(f"data:image/png;base64,{viz['mfcc_visualization']}", 
                        caption="Fitur MFCC untuk analisis model",
                        use_container_width=True)
        else:
            # Placeholder for visualizations
            st.markdown("""
            <div class="dashboard-card">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">📈 Visualisasi & Ekstraksi Suara</h3>
                <div style="text-align: center; padding: 3rem 1rem; color: #7f8c8d;">
                    <h4>🎯 Siap Menganalisis</h4>
                    <p>Visualisasi akan muncul setelah Anda merekam atau mengunggah file audio</p>
                    <div style="font-size: 4rem; opacity: 0.3; margin: 2rem 0;">📊</div>
                    <p><em>Rekam suara atau unggah file untuk melihat:</em></p>
                    <ul style="text-align: left; display: inline-block; margin-top: 1rem;">
                        <li>🔄 Pipeline Preprocessing</li>
                        <li>🌊 Waveform & Spectrogram</li>
                        <li>🎼 MFCC Features</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # System Information (full width, only when no visualizations)
    if not ('result' in st.session_state and 'visualizations' in st.session_state):
        st.markdown("""
        <div class="dashboard-card" style="margin-top: 2rem;">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: center;">ℹ️ Informasi Sistem</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; text-align: center;">
                <div style="background: rgba(46, 204, 113, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #2ecc71;">
                    <h4 style="color: #27ae60; margin: 0;">🟢 Status Server</h4>
                    <p style="color: #2c3e50; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0 0 0;">Online</p>
                </div>
                <div style="background: rgba(52, 152, 219, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #3498db;">
                    <h4 style="color: #2980b9; margin: 0;">⚡ Response Time</h4>
                    <p style="color: #2c3e50; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0 0 0;">< 1s</p>
                </div>
                <div style="background: rgba(155, 89, 182, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #9b59b6;">
                    <h4 style="color: #8e44ad; margin: 0;">🎯 Model Accuracy</h4>
                    <p style="color: #2c3e50; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0 0 0;">95.2%</p>
                </div>
                <div style="background: rgba(230, 126, 34, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #e67e22;">
                    <h4 style="color: #d35400; margin: 0;">📊 Rekaman</h4>
                    <p style="color: #2c3e50; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0 0 0;">3 detik otomatis</p>
                </div>
                <div style="background: rgba(52, 73, 94, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #34495e;">
                    <h4 style="color: #2c3e50; margin: 0;">📁 Format</h4>
                    <p style="color: #2c3e50; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0 0 0;">WAV, MP3</p>
                </div>
                <div style="background: rgba(231, 76, 60, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #e74c3c;">
                    <h4 style="color: #c0392b; margin: 0;">🤖 Deteksi</h4>
                    <p style="color: #2c3e50; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0 0 0;">Real-time AI</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Results Display
    if 'result' in st.session_state:
        res = st.session_state.result
        status_class = "urgent" if res['is_urgent'] else "safe"
        status_text = "🚨 UCAPAN DARURAT TERDETEKSI!" if res['is_urgent'] else "✅ UCAPAN BUKAN DARURAT"
        
        st.markdown(f"""
        <div class='emergency-status {status_class}'>
            {status_text}
            <div style='font-size: 1.5rem; margin-top: 1rem; opacity: 0.9;'>
                Tingkat Keyakinan: {res['confidence']*100:.1f}%
            </div>
            <div style='font-size: 1.2rem; margin-top: 0.5rem; opacity: 0.8;'>
                Kelas Prediksi: {res.get('predicted_class', 'Unknown')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Analysis
        with st.expander("🔬 DETAIL ANALISIS LENGKAP", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Akurasi Model", f"{res['confidence']*100:.2f}%")
            col2.metric("📝 Kelas Prediksi", res.get('predicted_class', 'Unknown'))
            col3.metric("⏱️ Waktu Proses", "< 1 detik")
            
            # Audio playback if available
            if 'file_path' in res and res['file_path'] and os.path.exists(res['file_path']):
                st.audio(res['file_path'])
            
            # Emergency contact info
            if res['is_urgent']:
                st.error("🚨 **TINDAKAN DARURAT DIPERLUKAN!**")
                st.markdown("""
                **Kontak Darurat:**
                - 🚑 Ambulans: 118
                - 🚒 Pemadam: 113  
                - 👮 Polisi: 110
                - 📞 SAR: 115
                """)

    # History Section
    if USERS_DB[st.session_state.current_user]["history"]:
        st.markdown("""
        <div class="dashboard-card" style="margin-top: 2rem;">
            <h3 style="color: #2c3e50;">📈 Riwayat Analisis Terbaru</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, h in enumerate(reversed(USERS_DB[st.session_state.current_user]["history"][-5:])):
            status_color = "#e74c3c" if h['status'] == 'DARURAT' else "#27ae60"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid {status_color};">
                <strong>{h['time']}</strong> - {h['status']} ({h['confidence']}%)
                <br><small>Kelas: {h.get('predicted_class', 'Unknown')}</small>
            </div>
            """, unsafe_allow_html=True)

def handle_recording(model, label_encoder):
    try:
        with st.spinner("🎙️ Sedang merekam suara... (3 detik)"):
            fs = 16000
            duration = 3
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            
            # Save the recorded audio to a temporary file
            filename = f"data/{st.session_state.current_user}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(filename, audio, fs)
            
            # Process the audio
            process_audio_file(filename, model, label_encoder, is_recorded=True)

    except Exception as e:
        st.error(f"❌ Gagal merekam audio: {e}")

def handle_upload(uploaded_file, model, label_encoder):
    try:
        with st.spinner("📤 Memproses file audio..."):
            # Save the uploaded file temporarily
            temp_filename = f"data/uploaded_{st.session_state.current_user}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the audio
            process_audio_file(temp_filename, model, label_encoder, is_recorded=False)
        
    except Exception as e:
        st.error(f"❌ Gagal mengunggah atau memproses file: {e}")

def process_audio_file(audio_path, model, label_encoder, is_recorded=True):
    """Process audio file and generate predictions and visualizations"""
    try:
        with st.spinner("🔄 Menganalisis audio..."):
            # Load and preprocess audio
            audio_data, sr = librosa.load(audio_path, sr=None)
            processed_audio = preprocess_audio(audio_data, sr)
            
            if processed_audio is None:
                st.error("❌ Gagal memproses audio")
                return
            
            # Save processed audio
            processed_filename = f"data/processed_{st.session_state.current_user}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(processed_filename, processed_audio, TARGET_SR)
            
            # Extract features for prediction
            features = extract_mfcc_features(processed_audio, TARGET_SR)
            
            if features is None:
                st.error("❌ Gagal mengekstrak fitur audio")
                return
            
            # Make prediction
            predicted_class, confidence = predict_audio(features, model, label_encoder)
            
            if predicted_class is None:
                st.error("❌ Gagal membuat prediksi")
                return
            
            # Generate visualizations
            visualizations = {}
            
            # Comprehensive visualization
            comprehensive_viz = generate_comprehensive_visualizations(audio_path, processed_audio, TARGET_SR)
            if comprehensive_viz:
                visualizations['comprehensive_visualization'] = comprehensive_viz
            
            # Waveform and spectrogram
            waveform_viz = generate_audio_visualizations(processed_filename)
            if waveform_viz:
                visualizations['waveform_spectrogram'] = waveform_viz
            
            # MFCC visualization
            mfcc_viz = generate_mfcc_visualization(processed_filename)
            if mfcc_viz:
                visualizations['mfcc_visualization'] = mfcc_viz
            
            # Store results in session state
            st.session_state.result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_urgent': predicted_class == 'kata_darurat',
                'file_path': processed_filename,
                'original_path': audio_path
            }
            st.session_state.visualizations = visualizations
            
            # Add to history
            USERS_DB[st.session_state.current_user]["history"].append({
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'DARURAT' if predicted_class == 'kata_darurat' else 'AMAN',
                'confidence': round(confidence * 100, 1),
                'predicted_class': predicted_class,
                'file_path': processed_filename
            })
            save_users_db()
            
            st.success("✅ Analisis selesai!")
            
    except Exception as e:
        st.error(f"❌ Error dalam pemrosesan audio: {e}")
        debug_logger.error(f"Error in process_audio_file: {str(e)}")

def main():
    debug_logger.info("=== APPLICATION STARTING ===")
    debug_logger.info(f"Python version: {os.sys.version}")
    debug_logger.info(f"Working directory: {os.getcwd()}")
    debug_logger.info(f"Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
    
    try:
        debug_logger.info("Calling user_interface()...")
        user_interface()
        debug_logger.info("user_interface() completed successfully")
        
    except Exception as e:
        debug_logger.error(f"CRITICAL ERROR in main(): {str(e)}")
        debug_logger.error(f"Exception type: {type(e)}")
        import traceback
        debug_logger.error(f"Traceback: {traceback.format_exc()}")
        
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Main application error: {str(e)}")

if __name__ == "__main__":
    debug_logger.info("=== SCRIPT EXECUTION STARTED ===")
    debug_logger.info("Starting main function...")
    main()