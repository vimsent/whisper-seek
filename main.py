from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import requests
import os

app = FastAPI()

# Configurar CORS para permitir que GitHub Pages hable con tu PC
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Por seguridad, luego cambia "*" por tu dominio de GitHub Pages
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo de Whisper (se descarga la primera vez)
# "base" es rápido, "medium" o "large" son más precisos pero lentos.
model = whisper.load_model("base") 

@app.post("/procesar-reunion")
async def procesar_reunion(file: UploadFile = File(...)):
    # 1. Guardar el archivo temporalmente
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 2. Transcribir con Whisper
        print("Iniciando transcripción...")
        result = model.transcribe(temp_filename)
        transcription_text = result["text"]
        print("Transcripción completada.")

        # 3. Resumir con DeepSeek R1 (vía Ollama local)
        print("Enviando a DeepSeek...")
        prompt = f"Resume la siguiente reunión de forma ejecutiva, listando puntos clave y tareas pendientes:\n\n{transcription_text}"
        
        # Llamada a la API de Ollama (asumiendo puerto 11434 por defecto)
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "deepseek-r1", # Asegúrate de tener este nombre exacto en `ollama list`
            "prompt": prompt,
            "stream": False
        })
        
        summary = response.json()['response']

        return {
            "transcription": transcription_text,
            "summary": summary
        }

    finally:
        # Limpieza
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Para correrlo: uvicorn main:app --reload