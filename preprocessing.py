import os                                                           # Standard Python-Modul, für BS-Funktionen und Arbeit mit Dateisystem
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importiert den TextSplitter
from langchain_experimental.text_splitter import SemanticChunker    # für semantisches Chunking
from langchain.schema import Document                               # Document-Klasse, um Dokumente zu erstellen und zu verwalten
from langchain_community.vectorstores import Chroma                 # Chroma-Klasse, um Vektorspeicher zu erstellen und zu verwalten
from langchain_huggingface import HuggingFaceEmbeddings             # HuggingFaceEmbeddings-Klasse, um Text in Vektoren umzuwandeln
from config import EMBEDDING_MODEL_NAME                             # eigene Konfigurationen
from docling.document_converter import DocumentConverter            # aktueller PDF-Reader
from datetime import datetime
import re                                                           # wichtig für Textbereinigung


def clean_text_for_semantic_chunking(text):
    """
    Bereinigt extrahierten Text, um typische Probleme für den SemanticChunker zu beheben.
    - Entfernt überflüssige Zeilenumbrüche, besonders in und um Zitate.
    - Normalisiert Leerräume.
    """
    # Heilt Zeilenumbrüche in und um Klammern, die typisch für Zitate sind.
    # Beispiel: "(a. a. \nO., S. \n25)." wird zu "(a. a. O., S. 25)."
    text = re.sub(r'\((.*?)\)', lambda m: '(' + m.group(1).replace('\n', ' ').replace('  ', ' ') + ')', text)
    # Ersetzt mehrere Zeilenumbrüche durch einen einzigen
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Entfernt Zeilenumbrüche, die wahrscheinlich durch das Layout entstanden sind (Wort am Ende der Zeile)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text

def load_document(file_path):
    '''
    Diese Funktion lädt eine PDF-Datei mit Docling und macht daraus eine strukturierte Sammlung von Seiten, 
    bei der jede Seite als LangChain-Dokument verfügbar ist – mit Text + Metadaten.
    '''

    if not os.path.exists(file_path):                       # Überprüft, ob die Datei existiert
        print(f"Datei nicht gefunden: {file_path}")         # Wenn nicht, wird eine Fehlermeldung ausgegeben
        return []                                           # und eine leere Liste zurückgegeben
    
    try:
        converter = DocumentConverter()                     # Docling Converter erstellen
        result = converter.convert(file_path)               # PDF mit Docling konvertieren
        docling_doc = result.document                       # result.document ist das konvertierte Dokument (Überschriften, Bilder, Tabellen, Textblöcke --> siehe markdown-Datei)
        full_text = docling_doc.export_to_markdown()
        cleaned_text = clean_text_for_semantic_chunking(full_text)
    except Exception as e:
        print(f"Fehler beim Laden von {file_path} mit Docling: {e}")
        return []
            
    doc_name = os.path.splitext(os.path.basename(file_path))[0]     # Dokumentname ohne Pfad und Endung extrahieren
    doc_name = os.path.splitext(os.path.basename(file_path))[0]
    document = Document(
        page_content=cleaned_text,
        metadata={
            "document_name": doc_name,
            "source": file_path,
            "extraction_method": "docling_full_only",
            "page_number": 1,
            "total_pages": 1
        }
    )
    return [document]


def load_all_pdfs_in_folder():
    '''
    Lädt alle PDF-Dateien aus dem aktuellen Ordner.
    '''
    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]  # gibt alle pdf-Dateien aus dem Ordner (Texte müssen also im selben Ordner sein)
    
    if not pdf_files:
        print("❌ Keine PDF-Dateien im aktuellen Ordner gefunden.")
        return []
    
    print(f"📚 {len(pdf_files)} PDF-Dateien gefunden:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    all_documents = []      # Sammlung aller Seiten der PDFs
    
    for pdf_file in pdf_files:              # gehe jede pdf durch
        print(f"\n📖 Lade: {pdf_file}")
        documents = load_document(pdf_file)
        all_documents.extend(documents)     
    
    print(f"\n✅ Insgesamt {len(all_documents)} Seiten aus {len(pdf_files)} PDFs geladen")
    return all_documents


def split_text_semantic(document_list):    
    '''
    Diese Funktion erhält als Argument unsere vorher geladenen Texte und unterteilt die Dokumente semantisch in kleinere Textabschnitte (Chunks).
    Verwendet SemanticChunker für thematische Trennung und einen Fallback für zu große Chunks.
    '''
    
    # Embedding-Modell für semantische Analyse, den der Chunker braucht
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Semantic Chunker analysiert den Text und teilt Chunks nach semantischer Ähnlichkeit
    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,              # dafür braucht es das Embedding Modell
        # Diese beiden Zeilen sind die Anweisung, wann ein Cut gemacht werden soll
        breakpoint_threshold_type="percentile", # (Prozentrang) - Methode, wie die Trenn-Schwelle gefunden wird
        breakpoint_threshold_amount=90          # "Ignoriere die 90% der kleinsten Themenwechsel"
                                                # "setze einen Cut nur bei den 10% der stärksten Themenwechsel" 
    
    )
    
    print("🧠 Semantische Chunk-Erstellung läuft...")
    
    # Erst semantisch aufteilen
    semantic_chunks = semantic_splitter.split_documents(document_list)
    print(f"📊 Semantische Aufteilung ergab {len(semantic_chunks)} Chunks")

    # Statistiken anzeigen
    chunk_sizes = [len(chunk.page_content) for chunk in semantic_chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    print(f"📏 Durchschnittliche Chunk-Größe: {avg_size:.0f} Zeichen")
    print(f"📏 Größter Chunk: {max(chunk_sizes)} Zeichen")
    print(f"📏 Kleinster Chunk: {min(chunk_sizes)} Zeichen")
    
    return semantic_chunks


def create_vectordb(chunks, db_path):
    '''
    Erstellt eine neue Vektordatenbank aus den Chunks.
    '''
   
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME) 
    
    print(f"🛠️ Neue Vektordatenbank wird erstellt: {db_path}")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    vectordb.persist()
    
    print(f"✅ Vektordatenbank wurde gespeichert!")
    return vectordb

def debug_chunks(chunks, show_content=True):
    '''
    Debug-Funktion um zu sehen, wie die Chunks aussehen
    '''
    num_to_show = len(chunks)
    print(f"\n{'='*60}")
    print(f"DEBUG: Zeige alle {num_to_show} Chunks")
    print(f"{'='*60}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i+1:03d} ---")
        print(f"📏 Länge: {len(chunk.page_content)} Zeichen")
        print(f"📋 Metadata: {chunk.metadata}")
        
        if show_content:
            content = chunk.page_content.replace('\n', ' ').strip()
            print(f"📄 Inhalt: {content}")                          
        
        print("-" * 50)


def save_chunks_to_file_simple(chunks, filename="semantic_chunks_raw.txt"):
    '''
    Speichert alle semantischen Chunks unverändert in eine Datei.
    '''
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"SEMANTIC CHUNK REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Anzahl Chunks: {len(chunks)}\n")
        f.write(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i+1:03d} ---\n")
            f.write(f"Länge: {len(chunk.page_content)} Zeichen\n")
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write(f"Inhalt:\n{chunk.page_content}\n")
            f.write(f"{'-'*60}\n\n")
    
    print(f"💾 Datei gespeichert: {filename}")


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

def save_docling_structure(docling_doc, doc_name):
    '''
    Speichert die Docling-Struktur des Dokuments in eine separate Datei.
    '''
    structure_dir = "docling_structure"
    if not os.path.exists(structure_dir):
        os.makedirs(structure_dir)
    
    structure_file = os.path.join(structure_dir, f"{doc_name}_docling_structure.md")
    
    with open(structure_file, 'w', encoding='utf-8') as f:
        f.write(f"# DOCLING STRUKTUR: {doc_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Gesamtes Dokument als Markdown exportieren
        try:
            full_markdown = docling_doc.export_to_markdown()
            f.write("## VOLLSTÄNDIGE MARKDOWN-STRUKTUR:\n")
            f.write("-" * 40 + "\n")
            f.write(full_markdown)
        except Exception as e:
            f.write(f"Fehler beim Markdown-Export: {e}\n")
    
    print(f"📋 Docling-Struktur gespeichert: {structure_file}")
    return structure_file


def main():
    print("🔄 Preprocessing mit SemanticChunker startet...")

    # Nur einen PDF-Dateinamen eingeben zum Testen
    file_path = "Das Gehirn.pdf"
    documents = load_document(file_path)
    
    if not documents:
        print("❌ Keine Dokumente geladen.")
        return
    
    chunks = split_text_semantic(documents)
    debug_chunks(chunks, show_content=True)  # zeigt Chunks im Terminal
    
    save_chunks_to_file_simple(chunks)
    print(f"✅ {len(chunks)} Chunks gespeichert.")

if __name__ == "__main__":
    main()