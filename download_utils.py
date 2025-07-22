import os
import gdown

def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists.")

def ensure_model_files():
    os.makedirs("model", exist_ok=True)

    files = {
        "model/best_model_full.h5": "17jM4JDTahLAJfUdcCrSbehs3gnJ3OpiD",
        "model/emotion_scaler.pkl": "1wGWaxlPaJ4pFfc5ZzZfwgmqn_XiH2Mjs",
        "model/emotion_encoder.pkl": "1g-Sj1QSR5neGhL9vD_-GgbmbHNw8ecsn"
    }

    for path, file_id in files.items():
        download_from_gdrive(file_id, path)
