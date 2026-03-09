"""
Parsea la respuesta cruda del agente a la estructura esperada por n8n:
mensajes (segmentos de texto), images (URLs Minio), images_count.
"""
import re


# Regex para URLs de imágenes Minio (n8nback)
IMAGE_REGEX = re.compile(
    r"https://minio\.biexedu\.com/n8nback/[A-Z0-9]+\.png",
    re.IGNORECASE,
)


def parse_response_to_structured(input_text: str) -> dict:
    """
    Aplica la misma lógica del nodo Code de n8n:
    - Extrae y deduplica URLs de imágenes Minio.
    - Limpia markdown del texto.
    - Segmenta por doble salto de línea.

    Retorna dict con: mensajes (list[str]), images (list[str]), images_count (int).
    """
    if not input_text or not isinstance(input_text, str):
        return {"mensajes": [], "images": [], "images_count": 0}

    # 1. Capturar todas las imágenes (pueden repetirse)
    images = list(IMAGE_REGEX.findall(input_text))
    # 2. Deduplicar
    images = list(dict.fromkeys(images))

    # 3. Remover URLs del texto
    cleaned_text = IMAGE_REGEX.sub("", input_text)

    # 4. Limpieza adicional de caracteres de formato Markdown
    cleaned_text = (
        cleaned_text.replace("**", "")
        .replace("*", "")
        .replace("[", "")
        .replace("]", "")
    )
    # Quitar guiones de listas tipo markdown (- texto)
    cleaned_text = re.sub(r"^\s*-\s*", "", cleaned_text, flags=re.MULTILINE)
    # Quitar encabezados markdown (#, ##, ###)
    cleaned_text = re.sub(r"^\s*#{1,6}\s*", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'(?<![a-zA-Z0-9/:\-])_([^_\n]+)_(?![a-zA-Z0-9/:\-])', r'\1', cleaned_text).strip()

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
