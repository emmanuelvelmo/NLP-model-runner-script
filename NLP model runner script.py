import os # Importa el módulo os para acceder a funciones del sistema operativo
import pathlib # Importa pathlib para manejo de rutas de archivos de manera moderna
import llama_cpp # Importa la librería para usar modelos de Llama en formato GGUF

# FUNCIONES
# Función para generar respuestas del modelo de lenguaje
def generar_respuesta(modelo_val, prompt_val):
    # Llama al modelo con los parámetros especificados y retorna la respuesta
    return modelo_val(
        prompt_val, # El texto de entrada (prompt) que se envía al modelo
        max_tokens = 1024, # Máximo número de tokens que puede generar en la respuesta
        temperature = 0.6, # Controla la aleatoriedad (0 = determinista, 1 = muy aleatorio)
        top_p = 0.9, # Muestreo nucleus: considera tokens que sumen hasta 90% de probabilidad
        top_k = 40, # Considera solo los 40 tokens más probables en cada paso
        repeat_penalty = 1.05, # Penaliza la repetición de palabras (>1 reduce repetición)
        min_p = 0.05, # Probabilidad mínima que debe tener un token para ser considerado
        frequency_penalty = 0.1, # Penaliza tokens que aparecen frecuentemente
        presence_penalty = 0.1, # Penaliza tokens que ya aparecieron en el texto
        stream = True, # Retorna la respuesta token por token (streaming)
        stop = ["User:", "Human:", "\n\n\n"], # Secuencias que detienen la generación
    )

# Función para cargar el modelo de lenguaje desde archivo
def cargar_modelo():
    carpeta_modelos = pathlib.Path("gguf")
    
    # Crear carpeta si no existe
    if not carpeta_modelos.exists():
        carpeta_modelos.mkdir(exist_ok = True)

        input('Place a .gguf model in "gguf" folder and press Enter...')
    
    # Buscar y cargar el modelo en un bucle infinito hasta encontrarlo
    while True:
        archivo_modelo = None

        # Buscar archivos .gguf en la carpeta
        for archivo_iter in carpeta_modelos.glob("*.gguf"):
            if archivo_iter.is_file():

                archivo_modelo = archivo_iter # Asignar el primer archivo encontrado

                break
        
        # Si se encontró un modelo, cargarlo
        if archivo_modelo:
            # Retorna una instancia del modelo Llama con configuración específica
            return llama_cpp.Llama(
                model_path = str(archivo_modelo), # Ruta del archivo del modelo
                n_ctx = 32768, # Tamaño del contexto (memoria del modelo)
                n_gpu_layers = -1, # Usar GPU para todas las capas (-1 = todas)
                n_threads = os.cpu_count() // 2, # Usar la mitad de los núcleos del CPU
                n_batch = 512, # Tamaño del lote para procesamiento
                use_mmap = True, # Usar memory mapping para cargar el modelo
                use_mlock = True, # Bloquear memoria para evitar swapping
                verbose = False, # No mostrar información detallada durante la carga
                f16_kv = True, # Usar precisión float16 para cache de key-value
                logits_all = False, # No calcular logits para todos los tokens
                vocab_only = False, # No cargar solo el vocabulario
                rope_scaling_type = llama_cpp.LLAMA_ROPE_SCALING_TYPE_LINEAR, # Tipo de escalado RoPE
            )
        else:
            input('Place a .gguf model in "gguf" folder and press Enter...')

# PUNTO DE PARTIDA
modelo_nlp = cargar_modelo() # Cargar el modelo al iniciar el programa
    
# Cargar modelo
modelo_dir = str(list(pathlib.Path("gguf").glob("*.gguf"))[0]) # Obtener la ruta del primer modelo encontrado

# Obtener solo el nombre del archivo del modelo
modelo_nombre = pathlib.Path(modelo_dir).name # Extraer el nombre del archivo sin la ruta completa

# Bucle principal del programa
while True:
    texto_entrada = input("Input: ").strip()
    
    # Verificar si la entrada está vacía
    if not texto_entrada:
        print() # Salto de línea

        continue # Retornar al inicio del while
    
    # Aplicar formato de prompt
    prompt_formateado = f"User: {texto_entrada}\nAssistant:"
    
    print() # Salto de línea
    
    # Generar respuesta token por token
    for token_salida in generar_respuesta(modelo_nlp, prompt_formateado):
        # Procesar cada token de la respuesta en streaming
        if 'choices' in token_salida:
            token_val = token_salida['choices'][0].get("text", "") # Extraer el texto del token

            print(token_val, end = '', flush = True) # Imprimir token inmediatamente sin salto de línea
    
    print("\n") # Doble salto de línea al terminar respuesta
