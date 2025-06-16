# 🚨 Emergency Voice Detection System

Sistem deteksi suara darurat berbasis AI dengan kemampuan rekam audio langsung dari browser.

## ✨ Fitur

- 🎙️ **Web Recording**: Rekam audio langsung dari browser menggunakan WebRTC
- 📁 **File Upload**: Upload file audio WAV/MP3
- 🤖 **AI Analysis**: Deteksi darurat menggunakan LSTM neural network
- 📊 **Visualizations**: Analisis lengkap sinyal audio (waveform, spectrogram, MFCC)
- ⚡ **Real-time**: Analisis instant setelah recording
- 🔒 **HTTPS Ready**: Compatible dengan cloud hosting

## 🚀 Live Demo

- **Streamlit Community Cloud**: [Coming Soon]
- **Local**: `streamlit run streamlit_app.py`

## 🛠️ Installation

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📋 Requirements

- Python 3.8+
- Modern browser (Chrome, Firefox, Safari, Edge)
- HTTPS untuk akses mikrofon (auto pada cloud hosting)

## 🎯 Cara Penggunaan

### 🎙️ Recording:
1. Klik "🎤 Mulai Rekam"
2. Allow microphone access
3. Speak clearly (3-5 seconds)
4. Click "⏹️ Stop Rekam"
5. Click "🔍 Analisis Audio"

### 📁 File Upload:
1. Click "Browse files"
2. Select WAV/MP3 file
3. Click "🔍 Analisis File"

## 🏗️ Architecture

- **Frontend**: Streamlit with WebRTC recording
- **Backend**: TensorFlow/Keras LSTM model
- **Audio Processing**: Librosa for MFCC feature extraction
- **Visualization**: Matplotlib for audio analysis charts

## 📦 Dependencies

- `streamlit` - Web app framework
- `streamlit-mic-recorder` - Web audio recording
- `tensorflow` - ML model inference
- `librosa` - Audio processing
- `matplotlib` - Visualizations
- `scikit-learn` - ML utilities

## 🌐 Deployment

### Streamlit Community Cloud:
1. Push to GitHub
2. Connect at share.streamlit.io
3. Deploy with `streamlit_app.py`

### Other Platforms:
- Railway, Render, Heroku compatible
- Requires HTTPS for microphone access

## 🔧 Configuration

No additional configuration needed. Model and encoder are generated automatically for demo purposes.

## 📝 License

MIT License

## 🤝 Contributing

Pull requests welcome! Please ensure HTTPS compatibility for recording features.

---

**Emergency Contacts (Indonesia):**
- 🚑 Ambulance: 118
- 🚒 Fire Dept: 113
- 👮 Police: 110
- 📞 SAR: 115