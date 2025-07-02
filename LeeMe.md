# Creamos entorno

    python -m venv ML_Eva4

# Dar permisos

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activamos Ambiente desde la raiz

    .\ML_Eva4\Scripts\activate

# Desactibar Ambiente desde la raiz

    cd .\ML_Eva4\Scripts
    deactivate

# Definir variables de entorno para FireBase

    $Env:GOOGLE_APPLICATION_CREDENTIALS = "ML-Eva4_key.json"

# Ver veriones de tags

    git tag -l 

# Eliminar carpeta desde PS

    Remove-Item -Path "ML_Eva4" -Recurse -Force
    Remove-Item -Path "dataset" -Recurse -Force

# Ejecuta entrenamiento con configuracion "config.yml"
    python scripts/train.py --config config/config.yml

    Esta linea guarda el modelo
    python scripts/train.py --config config/config.yml --model_path models/lstm_model.h5 --metrics_path models/metrics.json 