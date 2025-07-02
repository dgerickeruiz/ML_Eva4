# Para ejecutar API
# uvicorn api.main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import yaml
import pickle
import onnxruntime as ort
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Carga la configuración
with open('config/config.yml') as f:
    config = yaml.safe_load(f)

# Carga el tokenizer
with open('tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Carga el modelo Keras
model_keras = tf.keras.models.load_model('models/lstm_model.h5')

# Carga el modelo ONNX
onnx_session = ort.InferenceSession('models/lstm_model.onnx')

app = FastAPI(title="Sentiment Analysis API")

class TextRequest(BaseModel):
    text: str

@app.post("/predict/keras")
def predict_keras(data: TextRequest):
    seq = tokenizer.texts_to_sequences([data.text])
    padded = pad_sequences(seq, maxlen=config['max_len'], padding='post', truncating='post')
    pred = model_keras.predict(padded)[0][0]
    return {
        "input": data.text,
        "probability": float(pred),
        "sentiment": int(pred > 0.5)
    }

@app.post("/predict/onnx")
def predict_onnx(data: TextRequest):
    seq = tokenizer.texts_to_sequences([data.text])
    padded = pad_sequences(seq, maxlen=config['max_len'], padding='post', truncating='post')
    # onnxruntime espera numpy array de tipo int32 o float32
    input_array = np.array(padded).astype(np.float32)  # Cambia a int32 si tu modelo ONNX lo requiere
    # Nombre de la entrada del modelo (puedes verlo con: onnx_session.get_inputs()[0].name)
    input_name = onnx_session.get_inputs()[0].name
    #pred = onnx_session.run(None, {input_name: input_array})[0][0][0]
    pred = onnx_session.run(None, {"input_layer": input_array})[0][0][0]
    return {
        "input": data.text,
        "probability": float(pred),
        "sentiment": int(pred > 0.5)
    }