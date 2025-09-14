import os # Importa el módulo os para acceder a funciones del sistema operativo
import pathlib # Importa pathlib para manejo de rutas de archivos de manera moderna
import llama_cpp # Importa la librería para usar modelos de Llama en formato GGUF

# FUNCIONES
# Función para formatear el historial de mensajes
def format_chat_template(historial_mensajes):
    # Inicializa una cadena vacía que contendrá el prompt formateado
    historial_formateado = ""
    
    # Itera sobre cada mensaje en el historial de conversación
    for mensajes_val in historial_mensajes:
        # Agrega cada mensaje el formato
        historial_formateado += f"<|im_start|>{mensajes_val['role']}\n{mensajes_val['content']}<|im_end|>\n"
    
    # Agrega el token de inicio para la respuesta del asistente al final del prompt
    historial_formateado += "<|im_start|>assistant\n"
    
    # Retorna el prompt completo formateado listo para ser enviado al modelo
    return historial_formateado

# Función para generar respuestas del modelo de lenguaje
def generar_respuesta(modelo_val, prompt_val):
    # Llama al modelo con los parámetros especificados y retorna la respuesta
    return modelo_val(
        prompt_val, # El texto de entrada (prompt) que se envía al modelo
        max_tokens = 1024, # Máximo número de tokens que puede generar en la respuesta
        temperature = 0.6, # Controla la aleatoriedad (0 = determinista, 1 = muy aleatorio)
        top_p = 0.95, # Muestreo nucleus: considera tokens que sumen determinado porcentaje de probabilidad
        top_k = 40, # Considera solo los 40 tokens más probables en cada paso
        repeat_penalty = 1.1, # Penaliza la repetición de palabras (>1 reduce repetición)
        min_p = 0.05, # Probabilidad mínima que debe tener un token para ser considerado
        frequency_penalty = 0.1, # Penaliza tokens que aparecen frecuentemente
        presence_penalty = 0.1, # Penaliza tokens que ya aparecieron en el texto
        stream = True, # Retorna la respuesta token por token (streaming)
        stop = ["</s>", "[INST]", "[/INST]", "<|im_end|>"], # Secuencias que detienen la generación
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
                n_threads = os.cpu_count(), # Usar todos los núcleos de la CPU
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

# Historial de mensajes
historial_mensajes = [{"role": "system", "content": "You're a helpful assistant; your answers are concise and precise."}]

# Bucle principal del programa
while True:
    texto_entrada = input("Input: ").strip()
    
    # Verificar si la entrada está vacía
    if not texto_entrada:
        print() # Salto de línea

        continue # Retornar al inicio del while
    
    # Agregar mensaje de usuario al historial
    historial_mensajes.append({"role": "user", "content": texto_entrada})
    
    # Aplicar formato de prompt usando el historial
    prompt_formateado = format_chat_template(historial_mensajes)
    
    print() # Salto de línea
    
    # Variable para acumular la respuesta completa
    respuesta_completa = ""
    
    # Generar respuesta token por token
    for token_salida in generar_respuesta(modelo_nlp, prompt_formateado):
        # Procesar cada token de la respuesta en streaming
        if 'choices' in token_salida:
            token_val = token_salida['choices'][0].get("text", "") # Extraer el texto del token

            print(token_val, end = '', flush = True) # Imprimir token inmediatamente sin salto de línea
    
            respuesta_completa += token_val
    
    # Agregar respuesta del asistente al historial
    historial_mensajes.append({"role": "assistant", "content": respuesta_completa.strip()})
        
    # Mantener un historial limitado para evitar sobrepasar el contexto
    if len(historial_mensajes) > 10: # Mantener máximo 5 intercambios (user + assistant)
        historial_mensajes = [historial_mensajes[0]] + historial_mensajes[-8:] # Conservar system prompt + últimos 4 intercambios
    
    print("\n") # Doble salto de línea al terminar respuesta
