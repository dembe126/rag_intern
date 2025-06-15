# Zentrale Konfigurationsdatei

# Embedding-Modell
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Pfad für die Datenbanken
DB_BASE_PATH = "./chroma_dbs"

# Name der spezifischen Datenbank, die wir verwenden
DB_NAME = "all_documents"

# LLM-Modell für die Antwortgenerierung
LLM_MODEL = "llama3.2:latest" # Wir können hier ein Standardmodell festlegen