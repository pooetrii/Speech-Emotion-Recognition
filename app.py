import streamlit as st
import numpy as np
import librosa
from joblib import load
from download_utils import ensure_model_files
from tensorflow.keras.models import load_model
import librosa.display
import matplotlib.pyplot as plt

ensure_model_files()

# Load scaler, encoder, dan model
scaler = load('model/emotion_scaler.pkl')
encoder = load('model/emotion_encoder.pkl')
model = load_model('model/best_model_full.h5')

EXPECTED_FEATURE_LENGTH = 15444
AUDIO_FILE = 'temp_audio.wav'

# Styling
st.markdown("""
<style>
.stApp { background-color: #01012b; color: white; }
.emotion-output {
    text-align: center; margin-top: 30px; padding: 20px; border-radius: 15px;
    background-color: #111132; color: #ffffff; font-size: 26px; font-weight: bold;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
.plot-container {
    background-color: #111132;
    padding: 15px;
    border-radius: 20px;
    box-shadow: 0 2px 15px rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

emotion_emoji_dict = {
    "neutral": "😐", "calm": "😌", "happy": "😃", "sad": "😔",
    "angry": "😡", "fear": "😨", "disgust": "🤢", "surprise": "😮"
}

# Feature Extraction & Augmentation
class FeatureExtractor:
    def __init__(self, frame_length=2048, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def zcr(self, data):
        return librosa.feature.zero_crossing_rate(data, frame_length=self.frame_length, hop_length=self.hop_length).flatten()

    def rmse(self, data):
        return librosa.feature.rms(y=data, frame_length=self.frame_length, hop_length=self.hop_length).flatten()

    def mfcc(self, data, sr, n_mfcc=13):
        return librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=self.hop_length).T.flatten()

    def mel_spectrogram(self, data, sr):
        mel = librosa.feature.melspectrogram(y=data, sr=sr, hop_length=self.hop_length)
        return librosa.power_to_db(mel).flatten()

    def extract_features(self, data, sr):
        return np.concatenate([
            self.zcr(data), self.rmse(data),
            self.mfcc(data, sr), self.mel_spectrogram(data, sr)
        ])

class DataAugmentation:
    @staticmethod
    def noise(data, noise_factor=0.005):
        noise_amp = noise_factor * np.random.uniform() * np.amax(data)
        return data + noise_amp * np.random.normal(size=data.shape[0])

    @staticmethod
    def pitch(data, sr, n_steps=4):
        return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

class AudioProcessor:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.augmenter = DataAugmentation()

    def get_features(self, path):
        try:
            data, sr = librosa.load(path, duration=2.5, offset=0.6)
        except:
            return []
        features = [self.extractor.extract_features(data, sr)]
        features.append(self.extractor.extract_features(self.augmenter.noise(data), sr))
        pitched = self.augmenter.pitch(data, sr)
        features.append(self.extractor.extract_features(pitched, sr))
        features.append(self.extractor.extract_features(self.augmenter.noise(pitched), sr))
        return np.array(features)

def emotion_classifier(file_path):
    processor = AudioProcessor()
    X = processor.get_features(file_path)
    
    if len(X) == 0:
        return None, None

    if X.shape[1] < EXPECTED_FEATURE_LENGTH:
        X = np.pad(X, ((0, 0), (0, EXPECTED_FEATURE_LENGTH - X.shape[1])), mode='constant')
    elif X.shape[1] > EXPECTED_FEATURE_LENGTH:
        X = X[:, :EXPECTED_FEATURE_LENGTH]

    X_scaled = scaler.transform(X).reshape((X.shape[0], X.shape[1], 1))
    predictions = model.predict(X_scaled)

    predicted_indices = np.argmax(predictions, axis=1)
    final_index = np.bincount(predicted_indices).argmax()

    # one-hot vector untuk inverse_transform
    one_hot = np.zeros((1, len(encoder.categories_[0])))
    one_hot[0, final_index] = 1

    predicted_label = encoder.inverse_transform(one_hot)[0][0]
    return predicted_label, predictions[0]

# ========== UI ==========
st.title("🗣️ Speech Emotion Recognition")

st.markdown("### 🎧 Upload File Audio ")

file_uploaded = st.file_uploader("Unggah file audio (.wav / .mp3)", type=["wav", "mp3"])
if file_uploaded:
    st.audio(file_uploaded)
    with open(AUDIO_FILE, 'wb') as f:
        f.write(file_uploaded.read())

    try:
        data, sr = librosa.load(AUDIO_FILE, sr=None)
        col1, col2 = st.columns(2)

        # Waveform
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 📈 Waveform")
            fig_wave, ax_wave = plt.subplots(figsize=(6, 3))
            librosa.display.waveshow(data, sr=sr, ax=ax_wave)
            ax_wave.set_title('Waveform')
            st.pyplot(fig_wave)
            plt.close(fig_wave)
            st.markdown('</div>', unsafe_allow_html=True)
    
        # Mel Spectrogram
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 🌈 Mel Spectrogram")
            mel_spec = librosa.feature.melspectrogram(y=data, sr=sr)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            fig_mel, ax_mel = plt.subplots(figsize=(6, 3))
            img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=ax_mel, cmap='magma')
            fig_mel.colorbar(img, ax=ax_mel, format='%+2.0f dB')
            st.pyplot(fig_mel)
            plt.close(fig_mel)
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.warning(f"Tidak dapat menampilkan visualisasi audio: {e}")

    if st.button("🔍 Mulai Prediksi"):
        emotion, _ = emotion_classifier(AUDIO_FILE)
        if emotion:
            emoji = emotion_emoji_dict.get(emotion, "🎧")
            st.markdown(f'<div class="emotion-output">Prediksi Emosi: {emotion} {emoji}</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Gagal memprediksi emosi.")
