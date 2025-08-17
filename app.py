import streamlit as st
import numpy as np
import librosa
from joblib import load
from download_utils import ensure_model_files
from tensorflow.keras.models import load_model
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageDraw
from matplotlib import gridspec

ensure_model_files()
def make_rounded_plot(fig, radius=30):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    image = Image.open(buf).convert("RGBA")
    return image
    
# Load scaler, encoder, dan model
scaler = load('model/emotion_scaler.pkl')
encoder = load('model/emotion_encoder.pkl')
model = load_model('model/best_model_full.h5')

EXPECTED_FEATURE_LENGTH = 9288
AUDIO_FILE = 'temp_audio.wav'

# Styling
st.markdown("""
<style>
.stApp { background-color: #01012b; color: white; }
.emotion-output {
    text-align: center; margin-top: 30px; padding: 20px; border-radius: 15px;
    background-color: #111132; color: #ffffff; font-size: 26px; font-weight: bold;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
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

    def mfcc(self, data, sr, n_mfcc=13, flatten=True):
        mfcc_features = librosa.feature.mfcc(
            y=data, sr=sr, n_mfcc=n_mfcc, hop_length=self.hop_length
        )
        return mfcc_features.T.flatten() if flatten else mfcc_features.T

    def mel_spectrogram(self, data, sr, n_mels=64):
        mel_features = librosa.feature.melspectrogram(
            y=data, sr=sr, n_mels=n_mels, hop_length=self.hop_length
        )
        return librosa.power_to_db(mel_features).flatten()

    def extract_features(self, data, sr):
        return np.concatenate([
            self.zcr(data), self.rmse(data),
            self.mfcc(data, sr), self.mel_spectrogram(data, sr)
        ])

class DataAugmentation:
    @staticmethod
    def noise(data, noise_min=0.01, noise_max=0.07):
        noise_factor = np.random.uniform(noise_min, noise_max)
        noise_amp = noise_factor * np.amax(data)
        return data + noise_amp * np.random.normal(size=data.shape[0])

    @staticmethod
    def pitch(data, sr, min_steps=-2, max_steps=2):
        n_steps = np.random.randint(min_steps, max_steps + 1)
        return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

class AudioProcessor:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.augmenter = DataAugmentation()

    def pad_or_trim(self, feat):
        if len(feat) < EXPECTED_FEATURE_LENGTH:
            return np.pad(feat, (0, EXPECTED_FEATURE_LENGTH - len(feat)), 'constant')
        elif len(feat) > EXPECTED_FEATURE_LENGTH:
            return feat[:EXPECTED_FEATURE_LENGTH]
        return feat

    def get_features(self, path):
        data, sr = librosa.load(path, sr=22050)
        f = self.extractor.extract_features(data, sr)
        f = self.pad_or_trim(f)
        return np.array([f])

def emotion_classifier(file_path):
    processor = AudioProcessor()
    X = processor.get_features(file_path)              # shape: (1, 9288)

    X_scaled = scaler.transform(X)
    X_final = np.expand_dims(X_scaled, axis=2)
    predictions = model.predict(X_final)
    predicted_label = encoder.inverse_transform(predictions)[0][0]
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
            st.markdown("### 📈 Waveform")
            fig_wave, ax_wave = plt.subplots(figsize=(6, 3))
            fig_wave.patch.set_facecolor('#111132')
            ax_wave.set_facecolor('#111132')
            librosa.display.waveshow(data, sr=sr, ax=ax_wave)
            ax_wave.set_xlabel('Time', color='white')     
            ax_wave.set_ylabel('Amplitudo', color='white')    
            ax_wave.tick_params(colors='white')               
            rounded_wave = make_rounded_plot(fig_wave)
            st.image(rounded_wave)
            plt.close(fig_wave)
    
        # Mel Spectrogram
        with col2:
            st.markdown("### 🌈 Mel Spectrogram")
            fig_mel = plt.figure(figsize=(6,3.3))
            fig_mel.patch.set_facecolor('#111132')
            # Gunakan gridspec agar colorbar tidak ganggu layout utama
            gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.1)
            ax_mel = fig_mel.add_subplot(gs[0])
            cb_ax = fig_mel.add_subplot(gs[1])
            ax_mel.set_facecolor('#111132')
            mel_spec = librosa.feature.melspectrogram(y=data, sr=sr)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=ax_mel, cmap='magma')
            fig_mel.colorbar(img, cax=cb_ax, format='%+2.0f dB')
            ax_mel.set_xlabel('Time', color='white')         
            ax_mel.set_ylabel('Hz', color='white') 
            ax_mel.tick_params(colors='white')
            cb_ax.tick_params(colors='white')
            cb_ax.yaxis.set_tick_params(labelsize=8)
            rounded_mel = make_rounded_plot(fig_mel)
            st.image(rounded_mel)
            plt.close(fig_mel)
    
    except Exception as e:
        st.warning(f"Tidak dapat menampilkan visualisasi audio: {e}")

    if st.button("🔍 Mulai Prediksi"):
        emotion, _ = emotion_classifier(AUDIO_FILE)
        if emotion:
            emoji = emotion_emoji_dict.get(emotion, "🎧")
            st.markdown(f'<div class="emotion-output">Prediksi Emosi: {emotion} {emoji}</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Gagal memprediksi emosi.")

        # Tampilkan 3 probabilitas teratas (dari baris prediksi tunggal)
        top_indices = predictions.argsort()[-3:][::-1]
        top_labels = [encoder.categories_[0][i] for i in top_indices]
        top_probs = [predictions[i] for i in top_indices]
        df_top = pd.DataFrame({
            "Emosi": top_labels,
            "Probabilitas": [f"{p:.3f}" for p in top_probs]
        })
        st.table(df_top)
