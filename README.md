Aqui irá algo, algún día, tal vez, no lo sé, el futuro es incierto y el tiempo es escaso.

01/07/2025

Contador de errores y frustaciones: IIII

Parece que volví jeje

> Para entrenar el modelo ejecuta lo siguiente:

    python scripts/train.py --config config/config.yml --model_path models/lstm_model.h5 --metrics_path models/metrics.json

> Para exportar el modelo en ONNX

    python exportaModeloOnnx.py

    y luego:

    python -m tf2onnx.convert --saved-model models/saved_model --output models/lstm_model.onnx --opset 13

> Para levantar API en modo dev

    uvicorn api.main:app --reload

