import cv2
import numpy as np
import os

def procesar_video(ruta_video, num_frames=16, tamano=(112, 112)):
    """
    Toma un video, extrae un número fijo de frames uniformes
    y los redimensiona a un tamaño estándar.
    """
    # 1. Abrir el archivo de video
    captura = cv2.VideoCapture(ruta_video)
    
    # Obtener el total de frames que tiene el video originalmente
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Error: No se pudo leer el video en {ruta_video}")
        return None

    # 2. Calcular los índices de los frames que vamos a extraer
    indices_frames = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames_procesados = []

    for idx in range(total_frames):
        ret, frame = captura.read()
        if not ret:
            break
        
        # Si el frame actual está en nuestra lista de seleccionados, lo guardamos
        if idx in indices_frames:
            # Redimensionar la imagen 
            frame_redimensionado = cv2.resize(frame, tamano)
            
            # Normalizar los píxeles 
            frame_normalizado = frame_redimensionado / 255.0
            
            frames_procesados.append(frame_normalizado)
            
        if len(frames_procesados) == num_frames:
            break

    captura.release()
    
    # Convertir la lista de imágenes en un array de NumPy
    video_array = np.array(frames_procesados)
    return video_array

if __name__ == "__main__":
    
    ruta_prueba = "data/videos_raw/videoTest.mp4" 
    
    if os.path.exists(ruta_prueba):
        print("Procesando video de prueba...")
        resultado = procesar_video(ruta_prueba)
        if resultado is not None:
            print(f"¡Éxito! El video ahora es una matriz con forma: {resultado.shape}")
            print("(Frames, Alto, Ancho, Canales de color)")
    else:
        print(f"Por favor, coloca un video de prueba en: {ruta_prueba}")