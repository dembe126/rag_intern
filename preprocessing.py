import os                                                           # Standard Python-Modul, hilft mit dem Dateisystem zu arbeiten
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importiert den TextSplitter
from langchain_experimental.text_splitter import SemanticChunker    # für semantisches Chunking
from langchain.schema import Document                               # Document-Klasse, um Dokumente zu erstellen und zu verwalten
from langchain_community.vectorstores import Chroma                 # Chroma-Klasse, um Vektorspeicher zu erstellen und zu verwalten
from langchain_huggingface import HuggingFaceEmbeddings             # HuggingFaceEmbeddings-Klasse, um Text in Vektoren umzuwandeln
from config import EMBEDDING_MODEL_NAME                             # eigene Konfigurationen
from docling.document_converter import DocumentConverter            # aktueller PDF-Reader
from datetime import datetime


def load_document(file_path):
    '''
    Diese Funktion lädt eine PDF-Datei mit Docling und macht daraus eine strukturierte Sammlung von Seiten, 
    bei der jede Seite als LangChain-Dokument verfügbar ist – mit Text + Metadaten.
    '''

    if not os.path.exists(file_path):                       # Überprüft, ob die Datei existiert
        print(f"Datei nicht gefunden: {file_path}")         # Wenn nicht, wird eine Fehlermeldung ausgegeben
        return []                                           # und eine leere Liste zurückgegeben
    
    converter = DocumentConverter()                         # Docling Converter erstellen
    result = converter.convert(file_path)                   # PDF mit Docling konvertieren
    docling_doc = result.document                           # result.document ist das konvertierte Dokument (Überschriften, Bilder, Tabellen, Textblöcke --> siehe markdown-Datei)
        
    doc_name = os.path.splitext(os.path.basename(file_path))[0]     # Dokumentname ohne Pfad und Endung extrahieren
    
    
    documents = []      # hier wird jede Seite als eigenes Document-Objekt gespeichert
    
    if hasattr(docling_doc, 'pages') and docling_doc.pages:     # hat das docling-Dokument überhaupt Seiten?
        for page_num, page in enumerate(docling_doc.pages, 1):  # gehe durch jede Seite
            try:
                page_text = page.export_to_markdown()           # wandle Inhalt in Makrdwon um
            except:
                page_text = str(page.text) if hasattr(page, 'text') else ""     # wenn das nicht klappt, dann rohen Text oder Notlösung "" (leer)
            
            if page_text.strip():                      # falls Seite Text enthält
                documents.append(Document(             # speichere als Document mit:
                    page_content=page_text,            # der eigentliche Text
                    metadata={ 
                        'document_name': doc_name,      # Dokumentname
                        'source': file_path,            # Datepfad
                        'extraction_method': 'docling',
                        'page_number': page_num,        # Seitenzahl
                        'total_pages': len(docling_doc.pages)   # Gesamtseitenanzahl
                    }
                ))
    else:                                               # falls es keine Seiten gibt, exportiere alles auf eine Seite
        full_text = docling_doc.export_to_markdown()
        documents.append(Document(
            page_content=full_text,
            metadata={
                'document_name': doc_name,
                'source': file_path,
                'extraction_method': 'docling',
                'page_number': 1,
                'total_pages': 1
            }
        ))

    print(f"📄 Das Dokument wurde erfolgreich mit Docling geladen")
    print(f"📑 {len(documents)} Seiten extrahiert")
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"📏 Extrahierte Textlänge: {total_chars} Zeichen")
    
    return documents            # gib die Liste von Document-Objekten zurück, also jede Seite


def load_all_pdfs_in_folder():
    '''
    Lädt alle PDF-Dateien aus dem aktuellen Ordner.
    '''
    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("❌ Keine PDF-Dateien im aktuellen Ordner gefunden.")
        return []
    
    print(f"📚 {len(pdf_files)} PDF-Dateien gefunden:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    all_documents = []
    
    for pdf_file in pdf_files:
        print(f"\n📖 Lade: {pdf_file}")
        documents = load_document(pdf_file)
        all_documents.extend(documents)
    
    print(f"\n✅ Insgesamt {len(all_documents)} Seiten aus {len(pdf_files)} PDFs geladen")
    return all_documents


def split_text_semantic(document_list, max_chunk_size=400):    
    '''
    Diese Funktion unterteilt die Dokumente semantisch in kleinere Textabschnitte (Chunks).
    Verwendet SemanticChunker für thematische Trennung und einen Fallback für zu große Chunks.
    '''
    
    # Embedding-Modell für semantische Analyse (gleich wie für Vektordatenbank)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Semantischer Chunker - teilt nach Bedeutung auf
    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,              # Korrekter Parameter-Name
        breakpoint_threshold_type="percentile",  # oder "standard_deviation" für aggressivere Trennung
        breakpoint_threshold_amount=85
    )
    
    # Fallback-Splitter für zu große semantische Chunks
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=50,                        # weniger Overlap, da schon semantisch getrennt
        length_function=len,
        add_start_index=True,
    )
    
    print("🧠 Semantische Chunk-Erstellung läuft...")
    
    # Erst semantisch aufteilen
    semantic_chunks = semantic_splitter.split_documents(document_list)
    print(f"📊 Semantische Aufteilung ergab {len(semantic_chunks)} Chunks")

    # DANN kleine Chunks filtern/kombinieren
    filtered_chunks = []
    for chunk in semantic_chunks:
        if len(chunk.page_content.strip()) >= 50:  # Mindestlänge
            filtered_chunks.append(chunk)
        else:
            # Zu kleine Chunks mit dem nächsten kombinieren
            if filtered_chunks:
                filtered_chunks[-1].page_content += " " + chunk.page_content
    
    # Dann zu große Chunks nochmal aufteilen
    final_chunks = []
    large_chunks_split = 0
    
    for chunk in semantic_chunks:
        if len(chunk.page_content) > max_chunk_size:
            # Chunk ist zu groß -> weiter aufteilen
            sub_chunks = fallback_splitter.split_documents([chunk])
            final_chunks.extend(sub_chunks)
            large_chunks_split += 1
        else:
            final_chunks.append(chunk)
    
    if large_chunks_split > 0:
        print(f"✂️ {large_chunks_split} große Chunks wurden zusätzlich aufgeteilt")
    
    print(f"✅ Finale Chunk-Anzahl: {len(final_chunks)}")
    
    # Chunk-Größen-Statistik
    chunk_sizes = [len(chunk.page_content) for chunk in final_chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    print(f"📏 Durchschnittliche Chunk-Größe: {avg_size:.0f} Zeichen")
    print(f"📏 Größter Chunk: {max(chunk_sizes)} Zeichen")
    print(f"📏 Kleinster Chunk: {min(chunk_sizes)} Zeichen")
    
    return final_chunks


def create_vectordb(chunks, db_path):
    '''
    Erstellt eine neue Vektordatenbank aus den Chunks.
    '''
   
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)  # Erstellt ein Embedding-Modell mit dem konfigurierten Modellnamen
    
    print(f"🛠️ Neue Vektordatenbank wird erstellt: {db_path}")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    vectordb.persist()
    
    print(f"✅ Vektordatenbank wurde gespeichert!")
    return vectordb

def debug_chunks(chunks, show_all=False, show_content=True, max_content_chars=300):
    '''
    Debug-Funktion um zu sehen, wie die Chunks aussehen
    '''
    num_to_show = len(chunks) if show_all else 3
    
    print(f"\n{'='*60}")
    print(f"DEBUG: Zeige {num_to_show} von {len(chunks)} Chunks")
    print(f"{'='*60}")
    
    for i, chunk in enumerate(chunks[:num_to_show]):
        print(f"\n--- CHUNK {i+1:03d} ---")
        print(f"📏 Länge: {len(chunk.page_content)} Zeichen")
        print(f"📋 Metadata: {chunk.metadata}")
        
        if show_content:
            content = chunk.page_content.replace('\n', ' ').strip()
            if len(content) > max_content_chars:
                print(f"📄 Inhalt: {content[:max_content_chars]}...")
            else:
                print(f"📄 Inhalt: {content}")
        
        print("-" * 50)


def save_chunks_to_file(chunks, filename="chunks_analysis.txt"):
    '''
    Speichert alle Chunks in eine Textdatei für detaillierte Analyse
    '''
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"CHUNK ANALYSIS REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Anzahl Chunks: {len(chunks)}\n")
        f.write(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"CHUNK {i+1:03d}\n")
            f.write(f"Länge: {len(chunk.page_content)} Zeichen\n")
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write(f"Inhalt:\n{chunk.page_content}\n")
            f.write(f"{'='*60}\n\n")
    
    print(f"💾 Alle Chunks wurden in '{filename}' gespeichert")


def save_chunks_by_document(all_chunks):
    '''
    Speichert Chunks getrennt nach Dokumenten in eigene Dateien
    '''
    # Erstelle einen Ordner für die Chunk-Analysen
    analysis_dir = "chunk_analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
        print(f"📁 Ordner '{analysis_dir}' erstellt")
    
    # Gruppiere Chunks nach Dokumenten
    docs_chunks = {}
    for chunk in all_chunks:
        doc_name = chunk.metadata.get('document_name', 'unknown')
        if doc_name not in docs_chunks:
            docs_chunks[doc_name] = []
        docs_chunks[doc_name].append(chunk)
    
    print(f"📊 Erstelle Chunk-Analysen für {len(docs_chunks)} Dokumente...")
    
    # Speichere für jedes Dokument eine eigene Datei
    for doc_name, chunks in docs_chunks.items():
        filename = os.path.join(analysis_dir, f"{doc_name}_chunks.txt")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"CHUNK ANALYSIS: {doc_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Anzahl Chunks: {len(chunks)}\n")
            f.write(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1:03d}\n")
                f.write(f"Länge: {len(chunk.page_content)} Zeichen\n")
                f.write(f"Metadata: {chunk.metadata}\n")
                f.write(f"Inhalt:\n{chunk.page_content}\n")
                f.write(f"{'='*60}\n\n")
        
        print(f"  ✅ {doc_name}: {len(chunks)} Chunks → {filename}")
    
    # Zusätzlich: Gesamtübersicht aller Dokumente
    overview_file = os.path.join(analysis_dir, "00_overview.txt")
    with open(overview_file, 'w', encoding='utf-8') as f:
        f.write(f"CHUNK OVERVIEW - ALLE DOKUMENTE\n")
        f.write(f"{'='*60}\n")
        f.write(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Gesamtanzahl Chunks: {len(all_chunks)}\n\n")
        
        for doc_name, chunks in docs_chunks.items():
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            
            f.write(f"📄 {doc_name}:\n")
            f.write(f"   - Chunks: {len(chunks)}\n")
            f.write(f"   - Ø Größe: {avg_size:.0f} Zeichen\n")
            f.write(f"   - Größter: {max(chunk_sizes) if chunk_sizes else 0} Zeichen\n")
            f.write(f"   - Kleinster: {min(chunk_sizes) if chunk_sizes else 0} Zeichen\n\n")
    
    print(f"📋 Übersicht erstellt: {overview_file}")
    print(f"🎯 Alle Chunk-Analysen im Ordner: {analysis_dir}/")
    
    return analysis_dir


def main():
    """
    Hauptfunktion für das Preprocessing - verarbeitet alle PDFs im Ordner.
    """
    print("🔄 Multi-PDF-Preprocessing mit semantischem Chunking startet...")
    
    # Datenbank-Pfad für alle PDFs
    db_path = "./chroma_dbs/all_documents_semantic"
    
    # Prüfen ob Datenbank schon existiert
    if os.path.exists(db_path):
        print(f"🗃️ Datenbank existiert bereits: {db_path}")
        print("♻️ Verwende bestehende Datenbank...")
        # Lade bestehende Datenbank
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectordb = Chroma(persist_directory=db_path, embedding_function=embedding_model)
        print(f"✅ Bestehende Datenbank geladen!")
        return vectordb

    
    # Alle PDFs laden
    print("📚 Alle PDFs werden geladen...")
    all_documents = load_all_pdfs_in_folder()
    
    if not all_documents:
        print("❌ Keine Dokumente geladen.")
        return
    
    # Text semantisch in Chunks aufteilen
    print("🧠 Text wird semantisch in Chunks aufgeteilt...")
    chunks = split_text_semantic(all_documents, max_chunk_size=250)
    
    # Automatische Chunk-Analyse: Erstelle Dateien für jedes Dokument
    print("\n🔍 Erstelle Chunk-Analysen...")
    analysis_dir = save_chunks_by_document(chunks)
    
    # Vektordatenbank erstellen
    print("\n🛠️ Vektordatenbank wird erstellt...")
    vectordb = create_vectordb(chunks, db_path)
    
    print(f"\n🎉 Preprocessing abgeschlossen!")
    print(f"📁 Chunk-Analysen: {analysis_dir}/")
    print(f"🗃️ Vektordatenbank: {db_path}")

if __name__ == "__main__":
    main()