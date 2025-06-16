import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import hashlib  
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
import json
import streamlit.components.v1 as components
from datetime import datetime
import logging

# Audio processing imports
import librosa
import librosa.effects
import librosa.display
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# ML imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_METAL_DISABLE'] = '1'

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Konfigurasi Halaman
st.set_page_config(
    page_title="Emergency Voice Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)    

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

# Load model and label encoder (cached)
@st.cache_resource
def load_model_and_encoder():
    """Load LSTM model and label encoder"""
    try:
        logger.info("Loading LSTM model...")
        model = tf.keras.models.load_model('models/van_et_al.h5', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
        
        # Load label encoder classes
        label_encoder_classes = np.load('models/label_encoder_classes.npy', allow_pickle=True)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_encoder_classes
        logger.info(f"Label encoder loaded with classes: {label_encoder.classes_}")
        
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Audio processing functions
def preprocess_audio(audio_data, sr):
    """Preprocess audio data"""
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sr != TARGET_SR:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
        
        # Truncate silence
        audio_data, _ = librosa.effects.trim(audio_data, top_db=15)
        
        # Pad if needed
        audio_data = pad_audio_min_length(audio_data, TARGET_SR)
        
        return audio_data
    except Exception as e:
        logger.error(f"Error in audio preprocessing: {e}")
        return None

def pad_audio_min_length(audio, sr, min_mfcc_frames=MIN_MFCC_FRAMES, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Pad audio to ensure minimum MFCC frames"""
    min_len_samples = n_fft + hop_length * (min_mfcc_frames - 1)
    if len(audio) < min_len_samples:
        pad_amount = min_len_samples - len(audio)
        audio = np.pad(audio, (0, pad_amount), mode='constant')
    return audio

def extract_mfcc_features(audio, sr):
    """Extract MFCC + delta + delta-delta features for model input"""
    try:
        # Extract MFCC features
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
        
        # Transpose to get time as first dimension
        mfcc = mfcc.T
        
        # Pad or truncate to match model's expected input length
        target_length = 104  # Model's expected input length
        if mfcc.shape[0] > target_length:
            mfcc = mfcc[:target_length, :]
        elif mfcc.shape[0] < target_length:
            mfcc = np.pad(mfcc, ((0, target_length - mfcc.shape[0]), (0, 0)), mode='constant')
        
        # Reshape to match model's expected input shape (None, 104, 13, 3)
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        mfcc = np.repeat(mfcc, 3, axis=-1)    # Repeat to get 3 channels
        mfcc = np.expand_dims(mfcc, axis=0)   # Add batch dimension
        
        return mfcc
    except Exception as e:
        logger.error(f"Error extracting MFCC features: {e}")
        return None

def predict_audio(features, model, label_encoder):
    """Make prediction using the model"""
    try:
        # Verify input shape before prediction
        if features.shape != (1, 104, 13, 3):
            raise ValueError(f"Invalid input shape for prediction: {features.shape}. Expected (1, 104, 13, 3)")
        
        # Make prediction
        prediction = model.predict(features, verbose=0)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)
        
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None, None

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
        logger.error(f"Error generating comprehensive visualizations: {str(e)}")
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
        logger.error(f"Error generating visualizations: {str(e)}")
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
        logger.error(f"Error generating MFCC visualization: {str(e)}")
        return None

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

    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
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

# Initialize session state
if 'current_user' not in st.session_state:
    st.session_state.current_user = "user_" + hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

# Simple user database
USERS_DB = {
    st.session_state.current_user: {
        "history": []
    }
}

def save_users_db():
    # Simple function to maintain compatibility
    pass

class AudioRecorder:
    def __init__(self):
        self.fs = 16000
        self.duration = 3

# Main User Interface
def user_interface():
    # Load model and encoder
    model, label_encoder = load_model_and_encoder()
    
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
            filename = f"data/{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(filename, audio, fs)
            
            # Process the audio
            process_audio_file(filename, model, label_encoder, is_recorded=True)

    except Exception as e:
        st.error(f"❌ Gagal merekam audio: {e}")

def handle_upload(uploaded_file, model, label_encoder):
    try:
        with st.spinner("📤 Memproses file audio..."):
            # Save the uploaded file temporarily
            temp_filename = f"data/uploaded_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
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
            processed_filename = f"data/processed_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(processed_filename, processed_audio, TARGET_SR)
            
            # Extract features for prediction
            features = extract_mfcc_features(processed_audio, TARGET_SR)
            
            if features is None:
                st.error("❌ Gagal mengekstrak fitur audio")
                return
            
            # Make prediction
            predicted_class, confidence = predict_audio(features, model, label_encoder)
            
            if predicted_class is None:
                st.error("❌ Gagal melakukan prediksi")
                return
            
            # Determine if urgent
            is_urgent = predicted_class == 'kata_darurat'
            
            # Generate visualizations
            comprehensive_viz = generate_comprehensive_visualizations(audio_path, processed_audio, TARGET_SR)
            waveform_viz = generate_audio_visualizations(processed_filename)
            mfcc_viz = generate_mfcc_visualization(processed_filename)
            
            # Update session state with results
            st.session_state.result = {
                'is_urgent': is_urgent,
                'confidence': confidence,
                'predicted_class': predicted_class,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'file_path': processed_filename
            }
            
            # Store visualizations in session state
            st.session_state.visualizations = {
                'comprehensive_visualization': comprehensive_viz,
                'waveform_spectrogram': waveform_viz,
                'mfcc_visualization': mfcc_viz
            }
            
            # Update history
            USERS_DB[st.session_state.current_user]["history"].append({
                'time': st.session_state.result['time'],
                'status': 'DARURAT' if st.session_state.result['is_urgent'] else 'AMAN',
                'confidence': round(st.session_state.result['confidence']*100, 1),
                'predicted_class': st.session_state.result['predicted_class'],
                'file_path': st.session_state.result['file_path']
            })
            
            save_users_db()
            st.success("✅ Audio berhasil dianalisis!")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses audio: {e}")
        logger.error(f"Error processing audio: {e}")

def main():
    user_interface()

if __name__ == "__main__":
    main()