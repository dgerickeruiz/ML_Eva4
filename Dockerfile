# imagen base
FROM tensorflow/tensorflow:2.19.0
# directorio de trabajo
WORKDIR /app

# archivos necesarios
# api
COPY api/main.py api/main.py
# dependencias
COPY docker-requirements.txt requirements.txt
# configuración de entrenamiento
COPY config config
# modelo
COPY models models
# tokenizador
COPY tokenizer tokenizer

# instalación de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Se expone el puerto del servicio
EXPOSE 8000

# comando de entrada
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]