# source .venv/bin/activate

# Einfache Hauptdatei - startet das komplette RAG-System mit Docling

import os
from config import DB_BASE_PATH, DB_NAME
from new_preprocess import OptimizedRAGPreprocessor  # Deine neue Docling-basierte Klasse
from new_retrieval import OptimizedRAGRetriever, setup_ollama_model  # Deine neuen Retrieval-Funktionen


def show_retrieved_chunks(chunks, show_metadata=False):
    """
    Zeigt die gefundenen Chunks zur Transparenz an - erweitert für Docling-Metadaten.
    """
    print(f"\n🔍 {len(chunks)} relevante Chunks gefunden:")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        doc_name = chunk.metadata.get('document_name', 'Unbekannt')
        
        # Seitenzahl aus Docling-Metadaten
        page_info = ""
        if 'page_numbers' in chunk.metadata and chunk.metadata['page_numbers']:
            pages = chunk.metadata['page_numbers']
            if isinstance(pages, list):
                if len(pages) == 1:
                    page_info = f" (Seite {pages[0]})"
                else:
                    page_info = f" (Seiten {'-'.join(map(str, sorted(pages)))})"
            else:
                page_info = f" (Seite {pages})"
        elif 'page_number' in chunk.metadata:
            page_info = f" (Seite {chunk.metadata['page_number']})"
        
        # Überschrift aus Docling-Metadaten
        heading_info = ""
        if 'headings' in chunk.metadata and chunk.metadata['headings']:
            headings = chunk.metadata['headings']
            if isinstance(headings, list) and headings:
                heading_info = f" | {headings[0]}"
            elif isinstance(headings, str):
                heading_info = f" | {headings}"
        
        # Content preview
        content_preview = chunk.page_content[:150] + "..." if len(chunk.page_content) > 150 else chunk.page_content
        
        print(f"\n📄 CHUNK {i} - {doc_name}{page_info}{heading_info}")
        print(f"📏 Länge: {len(chunk.page_content)} Zeichen")
        print(f"📝 {content_preview}")
        
        if show_metadata and chunk.metadata:
            print(f"🏷️ Metadaten: {chunk.metadata}")
        
        print("-" * 40)


def show_system_status(db_path, vectordb, model_name):
    """
    Zeigt den aktuellen System-Status an
    """
    print("\n📊 SYSTEM-STATUS")
    print("=" * 50)
    print(f"🗃️ Datenbank: {os.path.basename(db_path)}")
    print(f"🤖 LLM-Modell: {model_name}")
    print(f"🔍 Embedding: {os.path.basename(os.getenv('EMBEDDING_MODEL_NAME', 'Standard'))}")
    
    if vectordb:
        try:
            collection_count = vectordb._collection.count()
            print(f"📚 Chunks in DB: {collection_count}")
        except:
            print(f"📚 Chunks in DB: Verfügbar")
    
    print("=" * 50)


def interactive_qa_loop(retriever, vectordb, model_name, db_path):
    """
    Hauptschleife für interaktive Fragen mit erweiterten Features
    """
    print(f"\n💬 Interaktives Q&A gestartet!")
    print("💡 Verfügbare Kommandos:")
    print("  - 'status' : System-Informationen anzeigen")
    print("  - 'debug'  : Nächste Antwort mit Chunk-Details")
    print("  - 'help'   : Diese Hilfe anzeigen")
    print("  - 'exit'   : Programm beenden")
    print("-" * 60)
    
    debug_mode = False
    
    while True:
        question = input("\n❓ Deine Frage: ").strip()
        
        # Kommandos verarbeiten
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 RAG-System beendet!")
            break
        
        elif question.lower() == "help":
            print("\n📖 HILFE:")
            print("  - Stelle Fragen zu den geladenen PDF-Dokumenten")
            print("  - 'status' zeigt System-Informationen")
            print("  - 'debug' aktiviert Details für die nächste Frage")
            print("  - 'exit' beendet das Programm")
            continue
        
        elif question.lower() == "status":
            show_system_status(db_path, vectordb, model_name)
            continue
        
        elif question.lower() == "debug":
            debug_mode = True
            print("🔍 Debug-Modus für nächste Frage aktiviert!")
            continue
        
        if not question:
            continue
        
        print("\n🔄 Verarbeitung läuft...")
        
        # Chunks abrufen und anzeigen (deine bestehende Logik)
        retriever_engine = vectordb.as_retriever(search_kwargs={'k': 8})
        retrieved_chunks = retriever_engine.invoke(question)
        
        if debug_mode:
            show_retrieved_chunks(retrieved_chunks, show_metadata=True)
            debug_mode = False
        else:
            show_retrieved_chunks(retrieved_chunks, show_metadata=False)
        
        # Erweiterte Abfrage mit dem neuen Retriever
        response = retriever.query_with_enhanced_response(question)
        
        if "error" in response:
            print(f"❌ {response['error']}")
            continue
        
        # Antwort anzeigen
        print(f"\n💡 ANTWORT:")
        print("-" * 50)
        print(response['answer'])
        
        # Verbesserte Quellen anzeigen
        if response['sources']:
            stats = response['chunk_stats']
            print(f"\n📚 QUELLEN ({stats['total_chunks']} Chunks aus {stats['unique_documents']} Dokument(en)):")
            for source in response['sources']:
                print(f"  {source}")
        
        print("\n" + "="*60)


def main():
    """
    Startet das komplette optimierte RAG-System.
    """
    
    print("🚀 Optimiertes RAG-System mit Docling wird gestartet...")
    
    # 1. Datenbank-Pfad definieren
    db_path = os.path.join(DB_BASE_PATH, "optimized_rag_docling")
    
    # 2. Preprocessor für Docling-basierte Verarbeitung
    preprocessor = OptimizedRAGPreprocessor(max_tokens=400, overlap_tokens=50)
    
    # 3. Datenbank laden oder erstellen
    if os.path.exists(db_path):
        print(f"\n📂 Schritt 1: Bestehende Docling-Datenbank wird geladen...")
        
        # Retriever initialisieren und Datenbank laden
        retriever = OptimizedRAGRetriever()
        vectordb = retriever.load_vectordb(db_path)
        
        if vectordb is None:
            print("❌ Fehler beim Laden der Datenbank.")
            return
            
    else:
        print(f"\n🛠️ Schritt 1: Neue Docling-Datenbank wird erstellt...")
        print("📋 Alle PDFs im aktuellen Ordner werden verarbeitet...")
        
        # Neue Datenbank mit allen PDFs erstellen
        vectordb = preprocessor.process_all_pdfs(db_name="optimized_rag_docling")
        
        if vectordb is None:
            print("❌ Fehler beim Erstellen der Datenbank.")
            return
        
        # Retriever nach der Erstellung initialisieren
        retriever = OptimizedRAGRetriever()
        retriever.vectordb = vectordb
    
    # 4. Ollama-Modell einrichten (deine bestehende Funktion)
    print(f"\n🤖 Schritt 2: Ollama-Modell wird eingerichtet...")
    model = setup_ollama_model()  # Nutzt deine bewährte Modell-Auswahl
    if model is None:
        print("❌ Kein Modell verfügbar. Programm beendet.")
        return
    
    # 5. RAG-Chain mit dem neuen Retriever aufbauen
    print(f"\n⚙️ Schritt 3: Optimierte RAG-Chain wird aufgebaut...")
    qa_chain = retriever.setup_rag_chain(model, retrieval_k=6)
    if qa_chain is None:
        print("❌ Fehler beim Aufbau der RAG-Chain.")
        return
    
    # 6. System-Status anzeigen
    show_system_status(db_path, vectordb, model)
    
    # 7. Interaktive Q&A-Schleife (erweiterte Version)
    interactive_qa_loop(retriever, vectordb, model, db_path)


if __name__ == "__main__":
    main()