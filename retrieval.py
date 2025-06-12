# Diese Datei ist für das Suchen und Antworten zuständig

import os
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


'''
Lädt eine bestehende Vektordatenbank.
'''

def load_vectordb(db_path):
    if not os.path.exists(db_path):
        print(f"❌ Vektordatenbank nicht gefunden: {db_path}")
        return None
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"📂 Vektordatenbank wird geladen: {db_path}")
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    
    return vectordb



'''
Sucht in der Vektordatenbank nach ähnlichen Chunks.
'''

def search_similar_chunks(query, vectordb, top_k=10):
    similar_chunks = vectordb.similarity_search(
        query=query,
        k=top_k
    )
    
    print(f"🔍 {len(similar_chunks)} relevante Chunks gefunden für: '{query}'")
    return similar_chunks



'''
Sendet die Frage und relevante Chunks an das LLM.
'''

def ask_llm_with_context(question, chunks, model_name="llama3.2:1b"):
    # Kontext aus Chunks zusammenbauen
    context = ""
    for i, chunk in enumerate(chunks, 1):
        docname = chunk.metadata.get("document_name", "Unbekanntes Dokument")
        page = chunk.metadata.get("page", "?")
        
        context += f"### Abschnitt {i}\n"
        context += f"Dokument: {docname}\n"
        context += f"Seite: {page}\n"
        context += f"Inhalt:\n{chunk.page_content}\n\n"
    
    # Prompt für das LLM erstellen
    prompt = f"""Du bist ein Experte und hilfreicher Assistent. Beantworte die folgende Frage basierend auf den gegebenen Textabschnitten sehr präzise und vollständig. 
    Die Antwort sollte nur aus den Informationen bestehen, die in den Textabschnitten stehen. Wenn du die gegebene Frage nicht aus Basis der gegebenen Textabschnitte und des Kontextes beantworten kann, teile dies mit.

KONTEXT:
{context}

FRAGE: {question}

ANTWORT: Beantworte die Frage basierend auf den Textabschnitten und gib auch die Seitenzahl(en) an, auf die du dich beziehst."""

    # API-Anfrage an Ollama
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        print("🤖 LLM generiert Antwort...")
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['response']
        else:
            return f"❌ Fehler bei der LLM-Anfrage: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Ollama ist nicht erreichbar. Ist es gestartet?"
    except Exception as e:
        return f"❌ Fehler: {str(e)}"



'''
Führt eine komplette RAG-Anfrage durch.
'''

def rag_query(question, vectordb, model_name="llama3.2:1b"):
    chunks = search_similar_chunks(question, vectordb, top_k=10)        # 1. Ähnliche Chunks suchen
    answer = ask_llm_with_context(question, chunks, model_name)         # 2. LLM mit Kontext fragen
    return answer



'''
Hilft beim Setup von Ollama-Modellen.
'''

def setup_ollama_model(model_name=None):
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Ollama ist nicht erreichbar")
            return None
            
        models = response.json()
        model_names = [model['name'] for model in models['models']]
        
        if not model_names:
            print("❌ Keine Modelle in Ollama gefunden")
            print("🔧 Installiere zuerst ein Modell mit: ollama pull MODEL_NAME")
            return None
            
        if model_name is None:
            print("📋 Verfügbare Modelle:")
            for idx, name in enumerate(model_names, 1):
                print(f"{idx}. {name}")
                
            while True:
                choice = input("➡️ Welches Modell möchtest du verwenden? (Zahl eingeben): ")
                if choice.isdigit() and 1 <= int(choice) <= len(model_names):
                    selected_model = model_names[int(choice) - 1]
                    print(f"✅ Modell '{selected_model}' ausgewählt!")
                    return selected_model
                else:
                    print("⚠️ Bitte eine gültige Zahl eingeben.")
        
        elif model_name in model_names:
            print(f"✅ Modell '{model_name}' ist verfügbar!")
            return model_name
        else:
            print(f"❌ Modell '{model_name}' nicht gefunden.")
            print(f"📋 Verfügbare Modelle: {', '.join(model_names)}")
            print(f"🔧 Du kannst es installieren mit: ollama pull {model_name}")
            return None
            
    except Exception as e:
        print(f"❌ Fehler bei der Verbindung zu Ollama: {str(e)}")
        return None



'''
Lässt den Benutzer eine Datenbank auswählen.
'''

def select_database():
    db_base_dir = "./chroma_dbs"
    
    if not os.path.exists(db_base_dir):
        print(f"❌ Datenbankordner nicht gefunden: {db_base_dir}")
        return None
    
    # Alle Datenbankordner finden
    db_folders = [f for f in os.listdir(db_base_dir) if os.path.isdir(os.path.join(db_base_dir, f))]
    
    if not db_folders:
        print("❌ Keine Datenbanken gefunden.")
        print("💡 Führe zuerst preprocessing.py aus, um eine Datenbank zu erstellen.")
        return None
    
    print("📚 Verfügbare Datenbanken:")
    for idx, db_name in enumerate(db_folders, 1):
        print(f"{idx}. {db_name}")
    
    while True:
        choice = input("➡️ Welche Datenbank möchtest du verwenden? (Zahl eingeben): ")
        if choice.isdigit() and 1 <= int(choice) <= len(db_folders):
            selected_db = db_folders[int(choice) - 1]
            return os.path.join(db_base_dir, selected_db)
        else:
            print("⚠️ Bitte eine gültige Zahl eingeben.")



def main():
    """
    Hauptfunktion für das Retrieval - interaktive Fragen.
    """
    print("🔍 RAG-Retrieval startet...")
    
    # 1. Datenbank auswählen
    db_path = select_database()
    if db_path is None:
        return
    
    # 2. Datenbank laden
    vectordb = load_vectordb(db_path)
    if vectordb is None:
        return
    
    # 3. Modell auswählen
    selected_model = setup_ollama_model()
    if selected_model is None:
        print("❌ Modell nicht gefunden oder Ollama nicht erreichbar.")
        return
    
    # 4. Interaktive Fragen
    print("\n💬 Du kannst jetzt Fragen stellen. Tippe 'exit', um zu beenden.\n")
    
    while True:
        user_input = input("❓ Deine Frage: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Bis zum nächsten Mal!")
            break
        
        answer = rag_query(user_input, vectordb, selected_model)
        print("\n💡 Antwort:\n", answer)
        print("-" * 60)


if __name__ == "__main__":
    main()