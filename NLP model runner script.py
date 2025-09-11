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
        top_p = 0.95, # Muestreo nucleus: considera tokens que sumen determinado porcentaje de probabilidad
        top_k = 40, # Considera solo los 40 tokens más probables en cada paso
        repeat_penalty = 1.1, # Penaliza la repetición de palabras (>1 reduce repetición)
        min_p = 0.05, # Probabilidad mínima que debe tener un token para ser considerado
        frequency_penalty = 0.1, # Penaliza tokens que aparecen frecuentemente
        presence_penalty = 0.1, # Penaliza tokens que ya aparecieron en el texto
        stream = True, # Retorna la respuesta token por token (streaming)
        stop = ["\n\n"], # Secuencias que detienen la generación
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

----------------------------------------------------------------------------------------------------------------------------------------

import os
import pathlib
import llama_cpp
import time

# FUNCIONES
def format_chat_template(messages, template_type="llama-2"):
    """
    Formatea mensajes según diferentes templates
    messages: lista de diccionarios [{"role": "user", "content": "texto"}, ...]
    """
    
    if template_type == "llama-2":
        # Formato de Llama 2
        system_prompt = ""
        conversation = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                conversation.append(f"[INST] {msg['content']} [/INST]")
            elif msg["role"] == "assistant":
                conversation.append(f"{msg['content']} </s><s>")
        
        if system_prompt:
            formatted = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n" + "\n".join(conversation)
        else:
            formatted = "<s>" + "\n".join(conversation)
        
        # Asegurar que termina correctamente para la generación
        if not formatted.endswith("[/INST]"):
            formatted = formatted.rstrip('</s><s>') + "[/INST]"
        return formatted
    
    elif template_type == "chatml":
        # Formato ChatML
        formatted = ""
        for msg in messages:
            formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    elif template_type == "mistral":
        # Formato para Mistral
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted += f"{msg['content']} </s>"
        return formatted
    
    else:
        # Formato simple (backup)
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

def generar_respuesta(modelo_val, prompt_val):
    return modelo_val(
        prompt_val,
        max_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1,
        min_p=0.05,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stream=True,
        stop=["</s>", "[INST]", "[/INST]", "<|im_end|>"],
    )

def cargar_modelo():
    carpeta_modelos = pathlib.Path("gguf")
    if not carpeta_modelos.exists():
        carpeta_modelos.mkdir(exist_ok=True)
        input('Coloca un modelo .gguf en la carpeta "gguf" y presiona Enter...')

    while True:
        archivo_modelo = None
        for archivo_iter in carpeta_modelos.glob("*.gguf"):
            if archivo_iter.is_file():
                archivo_modelo = archivo_iter
                break

        if archivo_modelo:
            print(f"Cargando modelo: {archivo_modelo.name}")
            return llama_cpp.Llama(
                model_path=str(archivo_modelo),
                n_ctx=32768,
                n_gpu_layers=-1,
                n_threads=os.cpu_count(),
                n_batch=512,
                use_mmap=True,
                use_mlock=True,
                verbose=False,
                f16_kv=True,
                logits_all=False,
                vocab_only=False,
                rope_scaling_type=llama_cpp.LLAMA_ROPE_SCALING_TYPE_LINEAR,
            )
        else:
            input('Coloca un modelo .gguf en la carpeta "gguf" y presiona Enter...')

# PUNTO DE PARTIDA
def main():
    modelo_nlp = cargar_modelo()
    modelo_dir = str(list(pathlib.Path("gguf").glob("*.gguf"))[0])
    modelo_nombre = pathlib.Path(modelo_dir).name
    
    print(f"Modelo cargado: {modelo_nombre}")
    print("Sistema listo. Escribe 'quit' para salir.\n")
    
    # Historial de mensajes
    messages = [
        {
            "role": "system", 
            "content": "Eres un asistente útil, responde de manera concisa y precisa."
        }
    ]
    
    # Detectar tipo de modelo para elegir el template apropiado
    model_name = modelo_nombre.lower()
    if "llama" in model_name:
        template_type = "llama-2"
    elif "mistral" in model_name:
        template_type = "mistral"
    else:
        template_type = "chatml"
    
    print(f"Usando template: {template_type}\n")

    while True:
        texto_entrada = input("Tú: ").strip()
        
        if not texto_entrada:
            continue
            
        if texto_entrada.lower() in ['quit', 'exit', 'salir']:
            break

        # Agregar mensaje de usuario al historial
        messages.append({"role": "user", "content": texto_entrada})
        
        # Formatear prompt según el template
        prompt_formateado = format_chat_template(messages, template_type)
        
        print("\nAsistente: ", end="", flush=True)
        
        respuesta_completa = ""
        start_time = time.time()
        token_count = 0
        
        # Generar respuesta token por token
        for token_salida in generar_respuesta(modelo_nlp, prompt_formateado):
            if 'choices' in token_salida:
                token_val = token_salida['choices'][0].get("text", "")
                print(token_val, end='', flush=True)
                respuesta_completa += token_val
                token_count += 1

        # Agregar respuesta del asistente al historial
        messages.append({"role": "assistant", "content": respuesta_completa.strip()})
        
        # Mantener un historial limitado para evitar sobrepasar el contexto
        if len(messages) > 10:  # Mantener máximo 5 intercambios (user + assistant)
            messages = [messages[0]] + messages[-8:]  # Conservar system prompt + últimos 4 intercambios
        
        generation_time = time.time() - start_time
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0
        
        print(f"\n\n[Tokens: {token_count} | Tiempo: {generation_time:.2f}s | Tokens/s: {tokens_per_second:.1f}]")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
