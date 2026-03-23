"""
Parsea la respuesta cruda del agente a la estructura esperada por el frontend:
mensajes (segmentos de texto con Markdown preservado), images (URLs Minio), images_count.
"""
import re


# Regex para URLs de imágenes Minio (n8nback)
IMAGE_REGEX = re.compile(
    r"https://minio\.biexedu\.com/n8nback/[A-Z0-9]+\.png",
    re.IGNORECASE,
)


def parse_response_to_structured(input_text: str) -> dict:
    """
    - Extrae y deduplica URLs de imágenes Minio.
    - Limpia artefactos internos del LLM (REPORT_JSON, tags de imágenes inventados).
    - Preserva formato Markdown (el frontend lo renderiza).
    - Segmenta por doble salto de línea.

    Retorna dict con: mensajes (list[str]), images (list[str]), images_count (int).
    """
    if not input_text or not isinstance(input_text, str):
        return {"mensajes": [], "images": [], "images_count": 0}

    # 1. Capturar todas las imágenes (pueden repetirse)
    images = list(IMAGE_REGEX.findall(input_text))
    # 2. Deduplicar
    images = list(dict.fromkeys(images))

    # 3. Remover URLs de imágenes Minio del texto
    cleaned_text = IMAGE_REGEX.sub("", input_text)

    # 4. Limpiar artefactos internos del LLM (NO tocar formato Markdown)
    # Bloques REPORT_JSON que el LLM copia del SPM (son internos, no visibles)
    cleaned_text = re.sub(r'<REPORT_JSON>[\s\S]*?</REPORT_JSON>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'REPORT_JSON', '', cleaned_text)

    # Tags de imágenes inventados por el LLM (no son parte del sistema)
    cleaned_text = re.sub(r'IMAGES\s*\{[^}]*\}\s*/IMAGES', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\[IMAGES\]\[/IMAGES\]', '', cleaned_text)
    cleaned_text = re.sub(r'https://image\.pollinations\.ai/[^\s]*', '', cleaned_text)

    cleaned_text = cleaned_text.strip()

    # 5. Segmentar texto por doble salto de línea
    segments = [
        s.strip()
        for s in cleaned_text.split("\n\n")
        if s and s.strip()
    ]

    if not segments and cleaned_text:
        segments = [cleaned_text]

    return {
        "mensajes": segments,
        "images": images,
        "images_count": len(images),
    }
