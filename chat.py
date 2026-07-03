import urllib.request
import json
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "dbname": "boletinDB",
    "user": "postgres",
    "password": "1234",
    "host": "127.0.0.1",
    "port": "5433",
    "options": "-c client_encoding=UTF8 -c lc_messages=C"
}

MODELO_OLLAMA = "gemma2:2b"
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 6
MAX_TURNOS_MEMORIA = 3


def conectar_db():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except UnicodeDecodeError as e:
        print(f"⚠️ Postgres devolvió un error no legible (bytes: {e.object!r}).")
        return None
    except Exception as e:
        print(f"⚠️ No se pudo conectar a Postgres ({e}).")
        return None


def buscar_nivel1_fulltext(cursor, pregunta, limite):
    """Full-text search en español. Excluye páginas de sumario/índice (listan muchos
    números de decreto de pasada y podrían ganarle en ranking al contenido real)."""
    cursor.execute(
        """
        SELECT id, texto, nro_boletin, archivo, pagina, pagina_fin, tipo_extraccion,
               ts_rank(texto_busqueda, plainto_tsquery('spanish', %s)) AS rank
        FROM public.chunks
        WHERE texto_busqueda @@ plainto_tsquery('spanish', %s)
          AND NOT (texto ILIKE '%%SUMARIO%%' AND texto ILIKE '%%Decretos%%')
        ORDER BY rank DESC
        LIMIT %s;
        """,
        (pregunta, pregunta, limite)
    )
    return cursor.fetchall()


def buscar_nivel2_semantico(cursor, modelo_embeddings, pregunta, limite, excluir_ids):
    """Búsqueda vectorial con pgvector, para preguntas conceptuales que no comparten palabras exactas."""
    vector = modelo_embeddings.encode(pregunta, convert_to_numpy=True).tolist()
    if excluir_ids:
        cursor.execute(
            """
            SELECT id, texto, nro_boletin, archivo, pagina, pagina_fin, tipo_extraccion,
                   1 - (embedding <=> %s::vector) AS similitud
            FROM public.chunks
            WHERE embedding IS NOT NULL AND id != ALL(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (vector, list(excluir_ids), vector, limite)
        )
    else:
        cursor.execute(
            """
            SELECT id, texto, nro_boletin, archivo, pagina, pagina_fin, tipo_extraccion,
                   1 - (embedding <=> %s::vector) AS similitud
            FROM public.chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (vector, vector, limite)
        )
    return cursor.fetchall()


def fila_a_dict(fila):
    return {
        "id": fila[0], "texto": fila[1], "nro_boletin": fila[2], "archivo": fila[3],
        "pagina": fila[4], "pagina_fin": fila[5], "tipo_extraccion": fila[6]
    }


def buscar_fragmentos(cursor, modelo_embeddings, pregunta, top_k=TOP_K):
    """Combina full-text + semántico, sin duplicar resultados."""
    resultados = []
    ids_usados = set()

    for fila in buscar_nivel1_fulltext(cursor, pregunta, top_k):
        resultados.append(fila_a_dict(fila))
        ids_usados.add(fila[0])

    if len(resultados) < top_k:
        faltan = top_k - len(resultados)
        for fila in buscar_nivel2_semantico(cursor, modelo_embeddings, pregunta, faltan, ids_usados):
            resultados.append(fila_a_dict(fila))
            ids_usados.add(fila[0])

    return resultados


def preguntar_a_ollama(pregunta, contexto, historial):
    url = "http://localhost:11434/api/generate"

    bloque_historial = ""
    if historial:
        turnos = "\n".join(f"Usuario: {h['pregunta']}\nAsistente: {h['respuesta']}" for h in historial)
        bloque_historial = f"\nCONVERSACIÓN PREVIA (para dar contexto a preguntas de seguimiento):\n{turnos}\n"

    prompt_sistema = f"""Sos un asistente experto en análisis de Boletines Oficiales.
Tu tarea es responder la pregunta del usuario utilizando ÚNICAMENTE los fragmentos de los boletines oficiales provistos en el CONTEXTO.

Reglas estrictas:
1. Sé preciso, formal y cita textualmente si es necesario.
2. Si en el contexto no figura la respuesta o no estás seguro, decí amablemente: "No encontré información sobre ese tema en los boletines cargados". No inventes nada.
3. IMPORTANTE: cada PREGUNTA DEL USUARIO es un tema independiente y nuevo, salvo que use una referencia explícita a la conversación anterior (por ejemplo "y quién lo firmó?", "¿y ese decreto...?", "eso mismo pero..."). Si la pregunta actual no tiene ninguna palabra que la conecte con la conversación previa, IGNORÁ COMPLETAMENTE la conversación previa y respondé solo en base al CONTEXTO de boletines actual. No relaciones ni mezcles información de una pregunta anterior con la pregunta actual si no hay una conexión explícita.
{bloque_historial}
CONTEXTO DE LOS BOLETINES:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

RESPUESTA:"""

    payload = {"model": MODELO_OLLAMA, "prompt": prompt_sistema, "stream": False}
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_json = json.loads(response.read().decode('utf-8'))
            return res_json.get("response", "")
    except Exception as e:
        return f"\n❌ Error al conectar con Ollama ({e})."


def main():
    print("🔌 Conectando a la base de datos de Docker...")
    conn = conectar_db()
    if not conn:
        print("❌ No se puede continuar sin conexión a la base.")
        return
    print("✅ Conectado a Postgres correctamente.")
    cursor = conn.cursor()

    print("⏳ Cargando el modelo de embeddings...")
    modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS)

    print("\n🚀 ¡Buscador Híbrido (Postgres + pgvector) Listo!")
    print("Para salir, escribí 'salir'.\n")

    historial = []

    while True:
        usuario_input = input("👤 Tu pregunta: ")
        if usuario_input.strip().lower() == "salir":
            print("¡Nos vemos!")
            break
        if not usuario_input.strip():
            continue

        print("🔍 Buscando...")
        try:
            fragmentos = buscar_fragmentos(cursor, modelo_embeddings, usuario_input)
        except (psycopg2.InterfaceError, psycopg2.OperationalError):
            print("⚠️ Se perdió la conexión. Reconectando...")
            conn = conectar_db()
            if not conn:
                print("❌ No se pudo reconectar.")
                break
            cursor = conn.cursor()
            continue

        print("\n🧪 DEBUG - Fragmentos recuperados:")
        for r in fragmentos:
            rango_pag = f"{r['pagina']}" if r['pagina'] == r['pagina_fin'] else f"{r['pagina']}-{r['pagina_fin']}"
            print(f"   boletin={r['nro_boletin']} | archivo={r['archivo']} | pag={rango_pag} | "
                  f"tipo={r['tipo_extraccion']} | preview={r['texto'][:60]!r}")
        print()

        contexto_bloque = ""
        fuentes = []
        for i, r in enumerate(fragmentos):
            contexto_bloque += f"--- Fragmento {i+1} (Boletín Nro: {r['nro_boletin']}) ---\n{r['texto']}\n\n"
            rango_pag = f"{r['pagina']}" if r['pagina'] == r['pagina_fin'] else f"{r['pagina']}-{r['pagina_fin']}"
            fuentes.append(
                f"📌 [Fuente] Boletín Nro: {r['nro_boletin']} | Archivo: {r['archivo']} | Página: {rango_pag}"
            )

        respuesta_ia = preguntar_a_ollama(usuario_input, contexto_bloque, historial[-MAX_TURNOS_MEMORIA:])

        print("\n🤖 Respuesta de la IA:")
        print(respuesta_ia)
        print("\n📄 Documentación de respaldo utilizada:")
        for f in fuentes:
            print(f)
        print("-" * 60 + "\n")

        historial.append({"pregunta": usuario_input, "respuesta": respuesta_ia})

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
