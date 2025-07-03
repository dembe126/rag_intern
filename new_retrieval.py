# Diese Datei ist für das Suchen und Antworten zuständig

import os
import json
import requests
from langchain_community.vectorstores import Chroma                 # Datenbank
from langchain_huggingface import HuggingFaceEmbeddings             # Embedding-Modell
from config import EMBEDDING_MODEL_NAME, DB_BASE_PATH, LLM_MODEL    # eigene Konfigurationen
from langchain.prompts import PromptTemplate                        # Um Prompt-Vorlage für LLM zu definieren
from langchain_community.llms import Ollama                         # Ollama für lokale Nutzung von LLM
from langchain.chains import RetrievalQA
from typing import List, Dict, Any


class OptimizedRAGRetriever:
    """
    Optimierter RAG-Retriever mit verbesserter Metadaten-Verarbeitung
    """
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectordb = None    # wird später initialisiert
        self.qa_chain = None    # wird später initialisiert
        
    def load_vectordb(self, db_path: str):
        """
        Lädt eine bestehende Vektordatenbank.
        """
        # Prüfe, ob der Pfad existiert. Falls nicht --> Fehlerhinweis und Abbruch
        if not os.path.exists(db_path):
            print(f"❌ Vektordatenbank nicht gefunden: {db_path}")
            return None
        
        print(f"📂 Vektordatenbank wird geladen: {db_path}")
        
        # Versucht, die Chroma-Datenbank zu initialisieren und die Embeddings-Funktion zu verbinden
        try:
            self.vectordb = Chroma(
                persist_directory=db_path,
                embedding_function=self.embedding_model
            )
            
            # Testet, ob die Datenbank funktioniert
            collection_count = self.vectordb._collection.count()
            print(f"✅ Datenbank geladen: {collection_count} Dokumente")    # gibt die Anzahl gespeicherter Dokumente aus
            
            return self.vectordb        # Gibt die geladene Datenbank zurück
            
        # Fehlermeldung
        except Exception as e:
            print(f"❌ Fehler beim Laden der Datenbank: {e}")
            return None

    def search_similar_chunks(self, query: str, k: int = 10) -> List[Dict]:
        """
        Sucht k Chunks, die dem eingegebenen query am ähnlichsten sind. 
        Rückgabe ist eine Liste von Dictionaries mit Infos zu den gefundenen Chunks.
        """
        # keine Vektor-Datenbank geladen? -->Suche abbrechen und leere Liste zurückgeben
        if not self.vectordb:
            print("❌ Keine Datenbank geladen!")
            return []
        
        try:
            # Similarity search mit Scores
            similar_chunks = self.vectordb.similarity_search_with_score(
                query=query,
                k=k
            )
            
            print(f"🔍 {len(similar_chunks)} relevante Chunks gefunden für: '{query}'")
            
            # Erweiterte Chunk-Informationen
            enhanced_chunks = []                    # Leere Liste zum Speichern der erweiterten Chunk-Infos
            for doc, score in similar_chunks:       # Geht jeden gefundenen Chunk und dessen Score durch
                chunk_info = {                      # Erstellt dictionary mit:
                    'content': doc.page_content,    # Inhalt des Chunks
                    'metadata': doc.metadata,       # Metadaten
                    'similarity_score': score,      # Ähnlichkeitsscore
                    'length': len(doc.page_content) # Länge des Chunks
                }
                enhanced_chunks.append(chunk_info)  # füge alles dem dict hinzu
            
            return enhanced_chunks
            
        except Exception as e:
            print(f"❌ Fehler bei der Suche: {e}")
            return []

    def setup_rag_chain(self, model_name: str, retrieval_k: int = 8):
        """
        Richtet eine optimierte RetrievalQA-Chain ein
        """
        if not self.vectordb:
            print("❌ Keine Datenbank geladen!")
            return None
        
        # 1. LLM definieren
        llm = Ollama(model=model_name, temperature=0.1)  # Niedrige Temperatur für konsistente Antworten

        # 2. Retriever mit optimierten Parametern
        retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': retrieval_k}
        )

        # 3. Verbesserter Prompt für Docling-Metadaten
        prompt_template = """Du bist ein hilfreicher Assistent für die Dokumentenanalyse. 

Nutze die folgenden Textabschnitte, um die Frage zu beantworten. Die Textabschnitte stammen aus strukturiert verarbeiteten PDF-Dokumenten.

WICHTIGE REGELN:
- Antworte nur basierend auf den bereitgestellten Informationen
- Wenn keine passende Antwort vorhanden ist: "Ich konnte keine relevante Information in den Dokumenten finden."
- Gib am Ende IMMER die Quellen an
- Verwende die Metadaten (Seitenzahlen, Überschriften) für präzise Quellenangaben

KONTEXT-ABSCHNITTE:
{context}

FRAGE: {question}

ANTWORT:"""

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        # 4. RetrievalQA-Chain mit optimierten Parametern
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": False  # Reduziert Debug-Output
            }
        )
        
        print(f"✅ RAG-Chain mit {model_name} ist einsatzbereit!")
        print(f"🔍 Retrieval-Parameter: Top-{retrieval_k} Chunks")
        return self.qa_chain

    def format_sources(self, source_documents) -> str:
        """
        Formatiert Quellen-Informationen basierend auf Docling-Metadaten
        """
        sources_info = []
        seen_sources = set()
        
        for doc in source_documents:
            metadata = doc.metadata
            
            # Dokument-Info
            doc_name = metadata.get('document_name', 'Unbekanntes Dokument')
            
            # Seiten-Info (aus Docling-Metadaten)
            page_info = ""
            if 'page_numbers' in metadata and metadata['page_numbers']:
                pages = metadata['page_numbers']
                if isinstance(pages, list):
                    if len(pages) == 1:
                        page_info = f", Seite {pages[0]}"
                    else:
                        page_info = f", Seiten {'-'.join(map(str, sorted(pages)))}"
                else:
                    page_info = f", Seite {pages}"
            elif 'page_number' in metadata:
                page_info = f", Seite {metadata['page_number']}"
            
            # Überschriften-Info
            heading_info = ""
            if 'headings' in metadata and metadata['headings']:
                headings = metadata['headings']
                if isinstance(headings, list) and headings:
                    heading_info = f" (Kapitel: {headings[0]})"
                elif isinstance(headings, str):
                    heading_info = f" (Kapitel: {headings})"
            
            # Chunking-Methode
            method_info = ""
            if metadata.get('chunking_method') == 'docling_hybrid':
                method_info = " [Docling]"
            
            source_key = f"{doc_name}{page_info}"
            if source_key not in seen_sources:
                sources_info.append(f"📄 {doc_name}{page_info}{heading_info}{method_info}")
                seen_sources.add(source_key)
        
        return sources_info

    def query_with_enhanced_response(self, question: str) -> Dict[str, Any]:
        """
        Erweiterte Abfrage mit detaillierten Antwort-Informationen
        """
        if not self.qa_chain:
            return {"error": "RAG-Chain nicht initialisiert"}
        
        try:
            # Abfrage ausführen
            result = self.qa_chain.invoke({"query": question})
            
            # Quellen formatieren
            formatted_sources = self.format_sources(result['source_documents'])
            
            # Chunk-Statistiken
            chunk_stats = {
                'total_chunks': len(result['source_documents']),
                'avg_chunk_length': sum(len(doc.page_content) for doc in result['source_documents']) / len(result['source_documents']) if result['source_documents'] else 0,
                'unique_documents': len(set(doc.metadata.get('document_name', 'Unknown') for doc in result['source_documents']))
            }
            
            return {
                'answer': result['result'],
                'sources': formatted_sources,
                'source_documents': result['source_documents'],
                'chunk_stats': chunk_stats,
                'question': question
            }
            
        except Exception as e:
            return {"error": f"Fehler bei der Abfrage: {e}"}


def setup_ollama_model(model_name=None):
    """
    Hilft beim Setup von Ollama-Modellen (deine bestehende Funktion)
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Ollama ist nicht erreichbar")
            print("💡 Starte Ollama mit: ollama serve")
            return None
            
        models = response.json()
        model_names = [model['name'] for model in models['models']]
        
        if not model_names:
            print("❌ Keine Modelle in Ollama gefunden")
            return None
            
        if model_name is None:
            print("📋 Verfügbare Modelle:")
            for idx, name in enumerate(model_names, 1):
                # Markiere Gemma-Modelle
                marker = "⭐" if "gemma" in name.lower() else "  "
                print(f"{idx}. {marker} {name}")
                
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


def main():
    """
    Hauptfunktion für das optimierte Retrieval
    """
    print("🚀 Optimiertes RAG-Retrieval startet...")
    
    # Datenbank auswählen
    db_path = os.path.join(DB_BASE_PATH, "optimized_rag_docling")
    if db_path is None: 
        return
    
    # RAG-Retriever initialisieren
    retriever = OptimizedRAGRetriever()
    
    # Datenbank laden
    vectordb = retriever.load_vectordb(db_path)
    if vectordb is None: 
        return
    
    # Ollama-Modell auswählen (bevorzugt Gemma)
    selected_model = setup_ollama_model(LLM_MODEL)  # Verwendet den Standardwert aus config.py
    if selected_model is None: 
        return
    
    # RAG-Chain einrichten
    qa_chain = retriever.setup_rag_chain(selected_model, retrieval_k=10)
    if qa_chain is None: 
        return
    
    print("\n💬 Du kannst jetzt Fragen stellen!")
    print("💡 Tipps: Verwende 'info' für Debug-Info, 'exit' zum Beenden")
    print("-" * 60)
    
    while True:
        user_input = input("\n❓ Deine Frage: ").strip()
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Bis zum nächsten Mal!")
            break
        
        if user_input.lower() == "info":
            # Debug-Informationen
            print("\n📊 System-Informationen:")
            print(f"🗃️ Datenbank: {os.path.basename(db_path)}")
            print(f"🤖 Modell: {selected_model}")
            print(f"🔍 Embedding-Modell: {EMBEDDING_MODEL_NAME}")
            continue
        
        if not user_input:
            continue
        
        print("\n🔍 Suche läuft...")
        
        # Erweiterte Abfrage
        response = retriever.query_with_enhanced_response(user_input)
        
        if "error" in response:
            print(f"❌ {response['error']}")
            continue
        
        # Antwort anzeigen
        print(f"\n💡 Antwort:")
        print("-" * 40)
        print(response['answer'])
        
        # Quellen anzeigen
        if response['sources']:
            print(f"\n📚 Quellen ({response['chunk_stats']['total_chunks']} Chunks aus {response['chunk_stats']['unique_documents']} Dokument(en)):")
            for source in response['sources']:
                print(f"  {source}")
        
        # Optionale Chunk-Details
        show_details = input("\n🔍 Chunk-Details anzeigen? (j/n): ").lower()
        if show_details == 'j':
            print("\n📄 Verwendete Chunks:")
            for i, doc in enumerate(response['source_documents'][:3], 1):  # Nur erste 3
                print(f"\n--- Chunk {i} ---")
                print(f"Länge: {len(doc.page_content)} Zeichen")
                print(f"Inhalt: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    main()