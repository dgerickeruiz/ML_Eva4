import tensorflow as tf
import tf2onnx

# Carga tu modelo entrenado
model = tf.keras.models.load_model('models/lstm_model.h5')
model.export('models/saved_model')  # Esto crea la carpeta SavedModel
print("✅ Modelo guardado en formato SavedModel")

# Ejecutar en consola: python exportaModeloOnnx.py

# Convertir a ONNX 
# python -m tf2onnx.convert --saved-model models/saved_model --output models/lstm_model.onnx --opset 13