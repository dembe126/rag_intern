# Diese Datei ist für das Suchen und Antworten zuständig

import os
import requests
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME, DB_BASE_PATH, LLM_MODEL               # Importiert Konfigurationen
from langchain.prompts import PromptTemplate                                   # PromptTemplate-Klasse, um benutzerdefinierte Prompts zu erstellen
from langchain_community.llms import Ollama                                    # Ollama-Klasse, um mit Ollama-Modellen zu interagieren
from langchain.chains import RetrievalQA                                       # RetrievalQA-Klasse, um Fragen mit Kontext zu beantworten


def load_vectordb(db_path):
    '''
    Lädt eine bestehende Vektordatenbank.
    '''
    if not os.path.exists(db_path):
        print(f"❌ Vektordatenbank nicht gefunden: {db_path}")
        return None
    
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"📂 Vektordatenbank wird geladen: {db_path}")
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    
    return vectordb


def search_similar_chunks(query, vectordb, top_k=10):
    '''
    Sucht in der Vektordatenbank nach ähnlichen Chunks.
    '''
    similar_chunks = vectordb.similarity_search(
        query=query,
        k=top_k
    )
    
    print(f"🔍 {len(similar_chunks)} relevante Chunks gefunden für: '{query}'")
    return similar_chunks


def setup_rag_chain(vectordb, model_name):
    '''
    Richtet eine RetrievalQA-Chain ein, die Fragen beantwortet.
    '''
    # 1. LLM definieren
    llm = Ollama(model=model_name)

    # 2. Den Vektor-Speicher als "Retriever" definieren.
    # Der Retriever ist dafür zuständig, die relevanten Chunks zu holen.
    retriever = vectordb.as_retriever(search_kwargs={'k': 10})                  # Wir holen die Top k Chunks

    # 3. Prompt-Vorlage erstellen
    # Hier definieren wir, wie der Input für das LLM aussehen soll.
    prompt_template = """
Du bist ein hilfreicher Assistent. Nutze die folgenden Textabschnitte, um die Frage am Ende zu beantworten.
Die Antwort sollte sich ausschließlich auf die bereitgestellten Informationen stützen.
Wenn die Antwort nicht im Kontext enthalten ist, antworte mit: "Ich konnte keine Antwort in den Dokumenten finden."

Am Ende deiner Antwort gibst du die Quelle an, aus der du die Information hast. Du findest den Dokumentennamen und die Seitenzahl in den Metadaten jedes Kontext-Abschnitts. Formatiere sie so: "Quelle: [Dokumentname], Seite [Seitenzahl]".

Kontext:
{context}

Frage: {question}

Hilfreiche Antwort:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 4. Die RetrievalQA-Chain erstellen
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                     # "stuff" bedeutet, alle gefundenen Chunks werden in den Prompt "gestopft"
        retriever=retriever,
        return_source_documents=True,           # Wir wollen die Quellen zurückbekommen
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("✅ RAG-Chain ist einsatzbereit!")
    return qa_chain


def setup_ollama_model(model_name=None):
    '''
    Hilft beim Setup von Ollama-Modellen.
    '''
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


def select_database():
    '''
    Lässt den Benutzer eine Datenbank auswählen.
    '''
    db_base_dir = DB_BASE_PATH 
    
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
    
    # ... (Code zum Datenbank und Modell auswählen bleibt gleich) ...
    db_path = select_database()
    if db_path is None: return
    
    vectordb = load_vectordb(db_path)
    if vectordb is None: return
    
    selected_model = setup_ollama_model()
    if selected_model is None: return
    
    # Hier kommt die neue Logik!
    rag_chain = setup_rag_chain(vectordb, selected_model)
    
    print("\n💬 Du kannst jetzt Fragen stellen. Tippe 'exit', um zu beenden.\n")
    
    while True:
        user_input = input("❓ Deine Frage: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Bis zum nächsten Mal!")
            break
        
        # Die Chain mit der Frage aufrufen
        result = rag_chain.invoke({"query": user_input})
        
        # Ergebnis ausgeben
        print("\n💡 Antwort vom Modell:\n", result['result'])
        
        # Zuverlässige Quellen aus dem Retriever anzeigen
        print("\n📚 Folgende Quellen wurden zur Beantwortung herangezogen:")
        unique_sources = set()
        for doc in result['source_documents']:
            doc_name = doc.metadata.get('document_name', 'N/A')
            page_num = doc.metadata.get('page', 'N/A')
            # Zeige jede Quelle nur einmal an
            unique_sources.add(f"  - Dokument: {doc_name}, Seite: {page_num}")

        for source in sorted(list(unique_sources)):
            print(source)

        print("-" * 60)

if __name__ == "__main__":
    main()