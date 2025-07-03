from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import pickle
import onnxruntime as ort
import numpy as np
import re
from keras.utils import pad_sequences

# Carga la configuración
with open('config/config002.yml') as f:
    config = yaml.safe_load(f)

# Carga el tokenizer
with open('tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Carga el modelo ONNX
onnx_session = ort.InferenceSession('models/lstm_model.onnx')

# Declaración de API
app = FastAPI(title="Sentiment Analysis API")

# Declaración de petición
class TextRequest(BaseModel):
    text: str

# limpieza de texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

@app.post("/predict/onnx")
def predict_onnx(data: TextRequest):
    # limpieza
    text_clean = preprocess_text(data.text)
    # secuenciación
    seq = tokenizer.texts_to_sequences([text_clean])
    # padding
    padded = pad_sequences(seq, maxlen=config['max_len'], padding='post', truncating='post')
    # generación de input esperado
    input_array = np.array(padded).astype(np.float32)
    # obtención de nombre del input
    input_name = onnx_session.get_inputs()[0].name
    # predicción
    pred = onnx_session.run(None, {input_name: input_array})[0][0][0]
    # respuesta
    return {
        "input": data.text,
        "value": float(pred),
        "sentiment": int(pred > 0.5)
    }