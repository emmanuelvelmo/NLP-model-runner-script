import pathlib
import sys
import os
import llama_cpp

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
            # Configuración mejorada
            return llama_cpp.Llama(
                model_path=str(archivo_modelo),
                n_ctx=32768,  # Aumentado significativamente
                n_gpu_layers=-1,
                n_threads=os.cpu_count() // 2,  # Usa la mitad de cores disponibles
                n_batch=512,
                use_mmap=True,
                use_mlock=True,
                verbose=False,
                # Parámetros adicionales para mejor rendimiento
                f16_kv=True,  # Usar fp16 para key-value cache
                logits_all=False,  # Solo calcular logits para el último token
                vocab_only=False,
                rope_scaling_type=llama_cpp.LLAMA_ROPE_SCALING_TYPE_LINEAR,  # Para modelos con contexto extendido
            )
        else:
            input('Place a .gguf model in "gguf" folder and press Enter...')

def aplicar_formato_prompt(texto_entrada, modelo_name=""):
    """
    Aplica formato de prompt según el tipo de modelo
    Ajusta según tu modelo específico
    """
    # Para modelos tipo Llama/Mistral
    if any(name in modelo_name.lower() for name in ['llama', 'mistral', 'vicuna']):
        return f"<s>[INST] {texto_entrada} [/INST]"
    
    # Para modelos tipo ChatML
    elif any(name in modelo_name.lower() for name in ['openchat', 'dolphin']):
        return f"<|im_start|>user\n{texto_entrada}<|im_end|>\n<|im_start|>assistant\n"
    
    # Para modelos tipo Alpaca
    elif 'alpaca' in modelo_name.lower():
        return f"### Instruction:\n{texto_entrada}\n\n### Response:\n"
    
    # Formato genérico si no se reconoce el modelo
    else:
        return f"User: {texto_entrada}\nAssistant:"

def generar_respuesta(modelo, prompt, max_tokens=2048):
    """
    Genera respuesta con parámetros optimizados
    """
    return modelo(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,  # Reducido para más consistencia
        top_p=0.9,        # Ligeramente reducido
        top_k=40,
        repeat_penalty=1.05,  # Reducido para evitar repetición excesiva
        min_p=0.05,
        frequency_penalty=0.1,  # Nuevo parámetro para evitar repeticiones
        presence_penalty=0.1,   # Nuevo parámetro para diversidad
        stream=True,
        stop=["User:", "Human:", "\n\n\n"],  # Tokens de parada
    )

# PUNTO DE PARTIDA
try:
    modelo_llama = cargar_modelo_desde_gguf()
    
    # Detectar tipo de modelo (opcional)
    modelo_path = str(list(pathlib.Path("gguf").glob("*.gguf"))[0])
    modelo_name = pathlib.Path(modelo_path).name
    
    while True:
        texto_entrada = input("Input: ").strip()
        
        if not texto_entrada:
            continue
            
        if texto_entrada.lower() == 'exit':
            break
        
        # Aplicar formato de prompt
        prompt_formateado = aplicar_formato_prompt(texto_entrada, modelo_name)
        
        print() # Salto de línea
        
        # Generar respuesta con configuración mejorada
        for token_output in generar_respuesta(modelo_llama, prompt_formateado):
            if 'choices' in token_output:
                token = token_output['choices'][0].get("text", "")
                print(token, end='', flush=True)
        
        print("\n")  # Doble salto de línea tras generar
        
except KeyboardInterrupt:
    print("\n\nChat interrumpido por el usuario.")
except Exception as e:
    print(f"Error loading model or running inference: {e}")
    print(f"Detalles: {type(e).__name__}")
