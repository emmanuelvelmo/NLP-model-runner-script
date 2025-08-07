import pathlib
import sys
import multiprocessing
import llama_cpp

# CARGA DE MODELO GGUF
def cargar_modelo_desde_gguf():
    carpeta_modelos = pathlib.Path("gguf")
    
    if not carpeta_modelos.exists():
        carpeta_modelos.mkdir(exist_ok=True)
        input('Place a .gguf model in "gguf" folder and press Enter...')
    
    while True:
        archivo_modelo = None
        for archivo_iter in carpeta_modelos.glob("*.gguf"):
            if archivo_iter.is_file():
                archivo_modelo = archivo_iter
                break
        
        if archivo_modelo:
            return llama_cpp.Llama(
                model_path=str(archivo_modelo),
                n_ctx=2048,
                n_gpu_layers=-1,
                n_threads=multiprocessing.cpu_count(),
                n_batch=512,
                use_mmap=True,
                use_mlock=True,
                verbose=False
            )
        else:
            input('Place a .gguf model in "gguf" folder and press Enter...')

# PUNTO DE PARTIDA
try:
    modelo_llama = cargar_modelo_desde_gguf()
    while True:
        texto_entrada = input("Input: ").strip()
        
        if not texto_entrada:
            continue
        
        print()  # Línea en blanco antes de la respuesta
        
        # Generación token por token con parámetros originales
        for token_output in modelo_llama(
            texto_entrada,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=["Input:"],
            stream=True
        ):
            if 'choices' in token_output:
                token = token_output['choices'][0].get("text", "")
                print(token, end='', flush=True)
        
        print("\n")  # Salto de línea tras generar
        
except Exception as e:
    print("Error loading model or running inference:", e)
