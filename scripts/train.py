# scripts/train.py

import argparse
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
import wandb
#from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger
import pickle

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data():
    # 🚨 Modifica esta parte para que cargue tu dataset real
    df = pd.read_csv('dataset/dataset_procesado.csv')  # ejemplo
    texts = df['text_procesado'].values
    labels = df['label'].values
    return texts, labels

def preprocess_data(texts, max_vocab, max_len):
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded, tokenizer

def build_model(embedding_dim, units, max_vocab, max_len):
    model = keras.Sequential([
        #keras.layers.Embedding(input_dim=max_vocab, output_dim=embedding_dim, input_length=max_len),
        keras.layers.Embedding(input_dim=max_vocab, output_dim=embedding_dim),
        keras.layers.LSTM(units),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main(config_path, model_path, metrics_path):
    # 📌 1. Cargar configuración
    config = load_config(config_path)

    # 📌 2. Iniciar wandb
    wandb.init(project="ML_Eva4", entity="dgerickeruiz24", config=config)

    # 📌 3. Cargar y preprocesar datos
    texts, labels = load_data()
    padded, tokenizer = preprocess_data(texts, config['max_vocab'], config['max_len'])

    # 📌 4. Construir modelo
    model = build_model(config['embedding_dim'], config['units'], config['max_vocab'], config['max_len'])

    # 📌 5. Entrenar modelo con callback de wandb
    history = model.fit(
        padded,
        labels,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        #callbacks=[WandbCallback()]
        callbacks=[WandbMetricsLogger()]
    )

    # 📌 6. Guardar modelo entrenado
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # 📌 7. Guardar métricas finales
    metrics = {
        'loss': history.history['loss'][-1],
        'accuracy': history.history['accuracy'][-1],
        'val_loss': history.history['val_loss'][-1],
        'val_accuracy': history.history['val_accuracy'][-1]
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    # Guardar tokenizer
    with open('tokenizer/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # 📌 8. Finalizar wandb
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yml')
    parser.add_argument('--model_path', type=str, default='models/lstm_model.h5')
    parser.add_argument('--metrics_path', type=str, default='models/metrics.json')
    args = parser.parse_args()

    main(args.config, args.model_path, args.metrics_path)
