# source .venv/bin/activate

# Einfache Hauptdatei - startet das komplette RAG-System

import os
from config import DB_BASE_PATH, DB_NAME
from preprocessing import (
    load_document, 
    split_text_semantic, 
    create_vectordb,  
    
)
from retrieval import (
    load_vectordb, 
    setup_rag_chain, 
    setup_ollama_model
)

def show_retrieved_chunks(chunks):
    """
    Zeigt die gefundenen Chunks zur Transparenz an.
    """
    print(f"\n🔍 {len(chunks)} relevante Chunks gefunden:")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        doc_name = chunk.metadata.get('document_name', 'Unbekannt')
        content_preview = chunk.page_content[:150] + "..." if len(chunk.page_content) > 150 else chunk.page_content
        
        print(f"\n📄 CHUNK {i} - {doc_name}")
        print(f"📝 {content_preview}")
        print("-" * 40)


def main():
    """
    Startet das komplette System.
    """
    
    print("🚀 RAG-System wird gestartet...")
    
    # 1. Feste Datenbank für alle Dokumente
    db_path = os.path.join(DB_BASE_PATH, DB_NAME)
    
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
        chunks = split_text_semantic(all_documents)
        
        # Datenbank erstellen
        vectordb = create_vectordb(chunks, db_path)
    
    # 3. Ollama-Modell einrichten
    print(f"\n🤖 Schritt 2: Modell wird eingerichtet...")
    model = setup_ollama_model()
    if model is None:
        print("❌ Kein Modell verfügbar. Programm beendet.")
        return
    
    # 4. RAG-Chain aufbauen
    print(f"\n⚙️ Schritt 3: RAG-Chain wird aufgebaut...")
    rag_chain = setup_rag_chain(vectordb, model)

    # 5. Interaktive Fragen
    print(f"\n💬 Schritt 4: Du kannst jetzt Fragen zu allen PDFs stellen!")
    print("Tippe 'exit' um das Programm zu beenden.\n")
    
    while True:
        question = input("❓ Deine Frage: ")
        
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 RAG-System beendet!")
            break

        # Chunks holen und anzeigen
        retriever = vectordb.as_retriever(search_kwargs={'k': 30})
        retrieved_chunks = retriever.invoke(question)
        show_retrieved_chunks(retrieved_chunks)

        # Die Chain mit der Frage aufrufen
        result = rag_chain.invoke({"query": question})
        
        # Ergebnis und Quellen ausgeben
        print("\n💡 Antwort vom Modell:\n", result['result'])
        
        print("\n📚 Folgende Quellen wurden zur Beantwortung herangezogen:")
        unique_sources = set()
        for doc in result['source_documents']:
            doc_name = doc.metadata.get('document_name', 'N/A')
            # Zeige jede Quelle nur einmal an
            unique_sources.add(f"  - Dokument: {doc_name}")

        for source in sorted(list(unique_sources)):
            print(source)

        print("-" * 60)


if __name__ == "__main__":
    main()