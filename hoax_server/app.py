import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Menyembunyikan beberapa pesan log TensorFlow

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# ====================================================================
# INISIALISASI APLIKASI DAN MODEL (Hanya berjalan sekali saat startup)
# ====================================================================

print(">>> Memulai server Flask...")
app = Flask(__name__)
CORS(app)

# Variabel Global
MAX_SEQUENCE_LENGTH = 500
model = None
keras_tokenizer = None
ner_model = None
ner_tokenizer = None
device = None

# 1. Muat Model Hoax Classifier Utama (.keras)
try:
    model = tf.keras.models.load_model('model_final.keras')
    print(">>> ✓ Model Hoax Classifier (.keras) berhasil dimuat!")
except Exception as e:
    print(f">>> ❌ Gagal memuat model .keras: {e}")

# 2. Muat Keras Tokenizer (untuk teks LSTM)
try:
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        keras_tokenizer = tokenizer_from_json(data)
    print(">>> ✓ Keras Tokenizer (.json) berhasil dimuat!")
except Exception as e:
    print(f">>> ❌ Gagal memuat tokenizer.json: {e}")

# 3. Muat Model NER dari Hugging Face
try:
    model_name = "cahya/bert-base-indonesian-NER"
    ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ner_model.to(device)
    ner_model.eval()
    print(f">>> ✓ Model NER berhasil dimuat! Menggunakan device: {device}")
except Exception as e:
    print(f">>> ❌ Gagal memuat model NER: {e}")


# ====================================================================
# FUNGSI-FUNGSI HELPER (Dicuplik dari kelas Anda)
# ====================================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_ner_features(text):
    if ner_model is None or ner_tokenizer is None:
        return np.zeros((1, 6)) # Kembalikan array nol jika NER gagal dimuat

    try:
        inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = ner_model(**inputs)
        
        predicted_token_class_ids = outputs.logits.argmax(dim=-1)
        predicted_tokens_classes = [ner_model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
        
        person_count = sum(1 for label in predicted_tokens_classes if 'PER' in label and label.startswith('B-'))
        org_count = sum(1 for label in predicted_tokens_classes if 'ORG' in label and label.startswith('B-'))
        loc_count = sum(1 for label in predicted_tokens_classes if 'LOC' in label and label.startswith('B-'))
        total_entities = person_count + org_count + loc_count
        entity_density = total_entities / len(predicted_tokens_classes) if predicted_tokens_classes else 0
        misc_count = sum(1 for label in predicted_tokens_classes if label.startswith('B-') and not any(e in label for e in ['PER','ORG','LOC']))

        # Mengembalikan sebagai numpy array dengan shape (1, 6)
        return np.array([[person_count, org_count, loc_count, misc_count, total_entities, entity_density]], dtype=np.float32)
    except Exception as e:
        print(f"Error saat ekstraksi NER: {e}")
        return np.zeros((1, 6))

# ====================================================================
# ENDPOINT API (VERSI BARU DENGAN VALIDASI)
# ====================================================================

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or keras_tokenizer is None:
        return jsonify({'error': 'Model atau Tokenizer tidak berhasil dimuat di server'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Request harus menyertakan "text"'}), 400

    text = data['text']

    # ====================================================================
    # BLOK VALIDASI TOPIK
    # ====================================================================
    timnas_keywords = [
        'timnas', 'tim nasional', 'garuda', 'shin tae yong', 
        'sty', 'pssi', 'pemain timnas', 'sepakbola indonesia',
        'indonesia vs', 'vs indonesia','timnas', 'u-23', 'u23', 'u-20', 'tim nasional',
        'garuda', 'piala', 'aff', 'asia', 'kualifikasi', 'fifa', 'laga', 'pertandingan',
        'skuad', 'pssi', 'pelatih', 'shin', 'erick', 'thohir', 'klasemen',
        'stadion', 'gol', 'melawan', 'kemenangan', 'klasemen', 'grup', 'zona',
        'lolos', 'babak', 'final', 'semi-final', 'penyerang', 'bek', 'kiper','patrick kluivert', 'line-up',
        'elkan', 'egy', 'asnawi', 'marselino', 'rizky ridho', 'pratama', 'rafli', 'ernando'
    ]
    
    # Memeriksa apakah ada salah satu keyword di dalam teks (setelah diubah ke huruf kecil)
    if not any(keyword in text.lower() for keyword in timnas_keywords):
        return jsonify({
            'error': 'Topik tidak sesuai', 
            'message': 'Tolong masukkan berita tentang timnas'
        }), 400
    # ====================================================================
    # AKHIR DARI BLOK VALIDASI
    # ====================================================================


    # 1. Bersihkan teks
    cleaned_text = clean_text(text)

    # 2. Ekstrak Fitur NER
    ner_features = extract_ner_features(cleaned_text)

    # 3. Tokenisasi dan Padding teks untuk LSTM
    sequence = keras_tokenizer.texts_to_sequences([cleaned_text])
    text_features = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # 4. Lakukan Prediksi dengan DUA input
    try:
        prediction = model.predict([text_features, ner_features])
        
        # Output dari softmax adalah probabilitas untuk setiap kelas
        # [prob_kelas_0, prob_kelas_1]
        hoax_probability = prediction[0][1] # Asumsikan kelas 1 adalah 'hoax'
        
        return jsonify({
            'prediction_score': float(hoax_probability),
            'is_hoax': bool(hoax_probability > 0.5)
        })
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({'error': 'Terjadi kesalahan saat melakukan prediksi'}), 500

if __name__ == '__main__':
    app.run(debug=True)
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Menyembunyikan beberapa pesan log TensorFlow

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import json
# import re
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import torch

# # ====================================================================
# # INISIALISASI APLIKASI DAN MODEL (Hanya berjalan sekali saat startup)
# # ====================================================================

# print(">>> Memulai server Flask...")
# app = Flask(__name__)
# CORS(app)

# # Variabel Global
# MAX_SEQUENCE_LENGTH = 500
# model = None
# keras_tokenizer = None
# ner_model = None
# ner_tokenizer = None
# device = None

# # 1. Muat Model Hoax Classifier Utama (.keras)
# try:
#     model = tf.keras.models.load_model('model_final.keras')
#     print(">>> ✓ Model Hoax Classifier (.keras) berhasil dimuat!")
# except Exception as e:
#     print(f">>> ❌ Gagal memuat model .keras: {e}")

# # 2. Muat Keras Tokenizer (untuk teks LSTM)
# try:
#     with open('tokenizer.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         keras_tokenizer = tokenizer_from_json(data)
#     print(">>> ✓ Keras Tokenizer (.json) berhasil dimuat!")
# except Exception as e:
#     print(f">>> ❌ Gagal memuat tokenizer.json: {e}")

# # 3. Muat Model NER dari Hugging Face
# try:
#     model_name = "cahya/bert-base-indonesian-NER"
#     ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
#     ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ner_model.to(device)
#     ner_model.eval()
#     print(f">>> ✓ Model NER berhasil dimuat! Menggunakan device: {device}")
# except Exception as e:
#     print(f">>> ❌ Gagal memuat model NER: {e}")


# # ====================================================================
# # FUNGSI-FUNGSI HELPER (Dicuplik dari kelas Anda)
# # ====================================================================

# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\S+', '', text)
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def extract_ner_features(text):
#     if ner_model is None or ner_tokenizer is None:
#         return np.zeros((1, 6)) # Kembalikan array nol jika NER gagal dimuat

#     try:
#         inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = ner_model(**inputs)
        
#         predicted_token_class_ids = outputs.logits.argmax(dim=-1)
#         predicted_tokens_classes = [ner_model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
        
#         person_count = sum(1 for label in predicted_tokens_classes if 'PER' in label and label.startswith('B-'))
#         org_count = sum(1 for label in predicted_tokens_classes if 'ORG' in label and label.startswith('B-'))
#         loc_count = sum(1 for label in predicted_tokens_classes if 'LOC' in label and label.startswith('B-'))
#         total_entities = person_count + org_count + loc_count
#         entity_density = total_entities / len(predicted_tokens_classes) if predicted_tokens_classes else 0
#         misc_count = sum(1 for label in predicted_tokens_classes if label.startswith('B-') and not any(e in label for e in ['PER','ORG','LOC']))

#         # Mengembalikan sebagai numpy array dengan shape (1, 6)
#         return np.array([[person_count, org_count, loc_count, misc_count, total_entities, entity_density]], dtype=np.float32)
#     except Exception as e:
#         print(f"Error saat ekstraksi NER: {e}")
#         return np.zeros((1, 6))

# # ====================================================================
# # ENDPOINT API
# # ====================================================================

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None or keras_tokenizer is None:
#         return jsonify({'error': 'Model atau Tokenizer tidak berhasil dimuat di server'}), 500

#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({'error': 'Request harus menyertakan "text"'}), 400

#     text = data['text']

#     # 1. Bersihkan teks
#     cleaned_text = clean_text(text)

#     # 2. Ekstrak Fitur NER
#     ner_features = extract_ner_features(cleaned_text)

#     # 3. Tokenisasi dan Padding teks untuk LSTM
#     sequence = keras_tokenizer.texts_to_sequences([cleaned_text])
#     text_features = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

#     # 4. Lakukan Prediksi dengan DUA input
#     try:
#         prediction = model.predict([text_features, ner_features])
        
#         # Output dari softmax adalah probabilitas untuk setiap kelas
#         # [prob_kelas_0, prob_kelas_1]
#         hoax_probability = prediction[0][1] # Asumsikan kelas 1 adalah 'hoax'
        
#         return jsonify({
#             'prediction_score': float(hoax_probability),
#             'is_hoax': bool(hoax_probability > 0.5)
#         })
#     except Exception as e:
#         print(f"Error saat prediksi: {e}")
#         return jsonify({'error': 'Terjadi kesalahan saat melakukan prediksi'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
