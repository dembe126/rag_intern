# source .venv/bin/activate

# Einfache Hauptdatei - startet das komplette RAG-System

import os
from preprocessing import (
    load_document, 
    split_text, 
    create_vectordb, 
    get_db_path, 
    
)
from retrieval import (
    load_vectordb, 
    rag_query, 
    setup_ollama_model
)


def main():
    """
    Startet das komplette System.
    """
    print("🚀 RAG-System wird gestartet...")
    
    # 1. Feste Datenbank für alle Dokumente
    db_path = "./chroma_dbs/all_documents"
    
    # 2. Datenbank laden oder erstellen
    if os.path.exists(db_path):
        print(f"\n📂 Schritt 1: Bestehende Datenbank mit allen Dokumenten wird geladen...")
        vectordb = load_vectordb(db_path)
    else:
        print(f"\n🛠️ Schritt 1: Neue Datenbank wird für alle PDFs erstellt...")
        # Alle PDFs laden
        from preprocessing import load_all_pdfs_in_folder
        all_documents = load_all_pdfs_in_folder()
        
        if not all_documents:
            print("❌ Keine PDFs im Ordner gefunden. Programm beendet.")
            return
        
        # In Chunks aufteilen
        chunks = split_text(all_documents)
        
        # Datenbank erstellen
        vectordb = create_vectordb(chunks, db_path)
    
    # 3. Ollama-Modell einrichten
    print(f"\n🤖 Schritt 2: Modell wird eingerichtet...")
    model = setup_ollama_model()
    if model is None:
        print("❌ Kein Modell verfügbar. Programm beendet.")
        return
    
    # 4. Interaktive Fragen
    print(f"\n💬 Schritt 3: Du kannst jetzt Fragen zu allen PDFs stellen!")
    print("Das System durchsucht automatisch alle Dokumente und nennt dir die Quelle.")
    print("Tippe 'exit' um das Programm zu beenden.\n")
    
    while True:
        question = input("❓ Deine Frage: ")
        
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 RAG-System beendet!")
            break
        
        answer = rag_query(question, vectordb, model)
        print(f"\n💡 Antwort:\n{answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()