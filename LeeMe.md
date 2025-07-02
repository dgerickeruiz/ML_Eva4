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

    $Env:GOOGLE_APPLICATION_CREDENTIALS = "dlops_key.json"

# Ver veriones de tags

    git tag -l 

# Eliminar carpeta desde PS

    Remove-Item -Path "ML_Eva4" -Recurse -Force  