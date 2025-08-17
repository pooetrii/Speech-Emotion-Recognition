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
        "model/splitnonsmoterndm.h5": "1a-BMLMgQSVNTAN94YrNTVtFJKFajNuNQ",
        "model/emotion_scalersplitrndm.pkl": "1GNFyZD2kVQyswNFPNxuGRj3lCH-dcSz2",
        "model/emotion_encodersplitrndm.pkl": "1qwZuRihB-LDWx9yWY0hqheHFrNo30xTI"
    }

    for path, file_id in files.items():
        download_from_gdrive(file_id, path)
