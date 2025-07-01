import os                                                           # Standard Python-Modul für Betriebssystemfunktionen wie Dateipfad-Operationen
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importiert den TextSplitter, um den Text bzw. Dokumente in kleinere Abschnitte (Chunks) zu unterteilen
from langchain.schema import Document                                # Importiert die Document-Klasse, um Dokumente zu erstellen und zu verwalten
from langchain_community.vectorstores import Chroma                 # Importiert die Chroma-Klasse, um Vektorspeicher zu erstellen und zu verwalten
from langchain_huggingface import HuggingFaceEmbeddings               # Importiert die HuggingFaceEmbeddings-Klasse, um Text in Vektoren umzuwandeln
from config import EMBEDDING_MODEL_NAME, DB_BASE_PATH, DB_NAME               # Importiert Konfigurationen
from docling.document_converter import DocumentConverter            # Neuer docling Import


def load_document(file_path):
    '''
    Diese Funktion lädt eine PDF-Datei mit Docling und gibt den Inhalt als Liste von Dokumenten zurück.
    '''

    if not os.path.exists(file_path):                       # Überprüft, ob die Datei existiert
        print(f"Datei nicht gefunden: {file_path}")         # Wenn nicht, wird eine Fehlermeldung ausgegeben
        return []                                           # und eine leere Liste zurückgegeben
    
    converter = DocumentConverter()                         # Docling Converter erstellen

    result = converter.convert(file_path)                   # PDF mit Docling konvertieren
    docling_doc = result.document
        
    doc_name = os.path.splitext(os.path.basename(file_path))[0]     # Dokumentname ohne Pfad und Endung extrahieren
    
    full_text = docling_doc.export_to_markdown()            # Text aus Docling-Dokument extrahieren (als Markdown)

    # LangChain Document erstellen (ähnlich wie PyMuPDF, aber nur ein Document statt eins pro Seite)
    document = [Document(
        page_content=full_text,
        metadata={
            'document_name': doc_name,
            'source': file_path,
            'extraction_method': 'docling',
            'total_pages': len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 1
        }
    )]

    print(f"📄 Das Dokument wurde erfolgreich mit Docling geladen")
    print(f"📏 Extrahierte Textlänge: {len(full_text)} Zeichen")
    if hasattr(docling_doc, 'pages'):
        print(f"📑 Originalseiten: {len(docling_doc.pages)}")
    return document                                         # Gibt die Liste der Dokumente (Seiten) zurück.     


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


def split_text(document_list):    
    '''
    Diese Funktion unterteilt die Dokumente in kleinere Textabschnitte (Chunks).
    Behält sowohl Dokumentname als auch Seitenzahl in den Metadaten.
    '''
    text_splitter = RecursiveCharacterTextSplitter(             # wir erstellen eine Instanz des TextSplitters
        chunk_size=500,                                         # ein Dokument wird in Chunks von chunk_size Zeichen unterteilt
        chunk_overlap=150,                                       # Zwischen aufeinanderfolgenden Chunks gibt es eine Überlappung von chunk_overlap Zeichen (für semantische Ähnlichkeit) 
        length_function=len,                                    # die Länge eines Chunks wird mit der Standard-Längenfunktion gemessen
        add_start_index=True,                                   # fügt den Startindex des Chunks als Metadaten hinzu
    )

    chunks = text_splitter.split_documents(document_list)       # wir teilen die Dokumente in Chunks auf, wobei jedes Dokument eine Seite repräsentiert
    
    print(f"✂️ Das Dokument wurde in {len(chunks)} Chunks unterteilt.")     # Gibt die Anzahl der Chunks aus
    return chunks                                                          # Gibt die Liste der Chunks zurück


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

def debug_chunks(chunks, num_chunks_to_show=3):
    '''
    Debug-Funktion um zu sehen, wie die Chunks aussehen
    '''
    print(f"\n=== DEBUG: Erste {num_chunks_to_show} Chunks ===")
    for i, chunk in enumerate(chunks[:num_chunks_to_show]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Länge: {len(chunk.page_content)} Zeichen")
        print(f"Metadata: {chunk.metadata}")
        print(f"Inhalt: {chunk.page_content[:200]}...")


def main():
    """
    Hauptfunktion für das Preprocessing - verarbeitet alle PDFs im Ordner.
    """
    print("🔄 Multi-PDF-Preprocessing startet...")
    
    # Datenbank-Pfad für alle PDFs
    db_path = "./chroma_dbs/all_documents"
    
    # Prüfen ob Datenbank schon existiert
    if os.path.exists(db_path):
        print(f"⚠️ Datenbank existiert bereits: {db_path}")
        overwrite = input("Möchtest du sie überschreiben? (j/n): ")
        if overwrite.lower() not in ["j", "ja", "y", "yes"]:
            print("❌ Preprocessing abgebrochen.")
            return
    
    # Alle PDFs laden
    print("📚 Alle PDFs werden geladen...")
    all_documents = load_all_pdfs_in_folder()
    
    if not all_documents:
        print("❌ Keine Dokumente geladen.")
        return
    
    # Text in Chunks aufteilen
    print("✂️ Text wird in Chunks aufgeteilt...")
    chunks = split_text(all_documents)
    
    # Vektordatenbank erstellen
    print("🛠️ Vektordatenbank wird erstellt...")
    vectordb = create_vectordb(chunks, db_path)
    
    print(f"🎉 Preprocessing abgeschlossen! Alle PDFs in einer Datenbank: {db_path}")


if __name__ == "__main__":
    main()