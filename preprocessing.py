import os                                                           # Standard Python-Modul für Betriebssystemfunktionen wie Dateipfad-Operationen
from langchain_community.document_loaders import PyMuPDFLoader        # Importiert den PDFLoader aus dem langchain_community-Paket, um PDF-Dokumente zu laden
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importiert den TextSplitter, um den Text bzw. Dokumente in kleinere Abschnitte (Chunks) zu unterteilen
from langchain.schema import Document                                # Importiert die Document-Klasse, um Dokumente zu erstellen und zu verwalten
from langchain_community.vectorstores import Chroma                 # Importiert die Chroma-Klasse, um Vektorspeicher zu erstellen und zu verwalten
from langchain_huggingface import HuggingFaceEmbeddings               # Importiert die HuggingFaceEmbeddings-Klasse, um Text in Vektoren umzuwandeln
from config import EMBEDDING_MODEL_NAME, DB_BASE_PATH               # Importiert Konfigurationen


'''
Diese Hilfsfunktion erstellt einen passenden Pfadnamen für die Vektordatenbank basierend auf dem Namen der PDF-Datei.
'''

def get_db_path(file_path, base_dir=DB_BASE_PATH):
    filename = os.path.basename(file_path)                  # Extrahiert den Dateinamen aus dem vollständigen Pfad (z.B. "dokument.pdf" aus "/ordner/dokument.pdf")
    name, _ = os.path.splitext(filename)                    # Trennt den Dateinamen vom Dateityp (z.B. "dokument" von ".pdf")
    return os.path.join(base_dir, name.lower())             # Gibt den vollständigen Pfad zurück, wo die Vektordatenbank gespeichert werden soll (z.B. "./chroma_dbs/dokument")



'''
Diese Funktion lädt eine PDF-Datei und gibt den Inhalt als Liste von Dokumenten zurück.
'''

def load_document(file_path):

    if not os.path.exists(file_path):                       # Überprüft, ob die Datei existiert
        print(f"Datei nicht gefunden: {file_path}")         # Wenn nicht, wird eine Fehlermeldung ausgegeben
        return []                                           # und eine leere Liste zurückgegeben
    
    loader = PyMuPDFLoader(file_path)                         # Erstellt eine Instanz des PyPDFLoader mit dem angegebenen Dateipfad
    document = loader.load()                                # Ruft die load()-Methode auf, die das PDF liest → jede Seite der PDF wird zu einem einzelnen Text-Dokument in einer Liste
    
    doc_name = os.path.splitext(os.path.basename(file_path))[0]     # Dokumentname ohne Pfad und Endung extrahieren
    
    
    for doc in document:                                            # Dokumentname zu jedem Seiten-Dokument hinzufügen
        doc.metadata['document_name'] = doc_name
    
    print (f"📄 Das Dokument hat {len(document)} Seiten")   # Gibt die Anzahl der der geladenen Seiten im Dokument aus
    return document                                         # Gibt die Liste der Dokumente (Seiten) zurück.     



'''
Lädt alle PDF-Dateien aus dem aktuellen Ordner.
'''


def load_all_pdfs_in_folder():
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



'''
Diese Funktion unterteilt die Dokumente in kleinere Textabschnitte (Chunks).
Behält sowohl Dokumentname als auch Seitenzahl in den Metadaten.
'''

def split_text(document_list):    
    text_splitter = RecursiveCharacterTextSplitter(             # wir erstellen eine Instanz des TextSplitters
        chunk_size=500,                                         # ein Dokument wird in Chunks von chunk_size Zeichen unterteilt
        chunk_overlap=150,                                       # Zwischen aufeinanderfolgenden Chunks gibt es eine Überlappung von chunk_overlap Zeichen (für semantische Ähnlichkeit) 
        length_function=len,                                    # die Länge eines Chunks wird mit der Standard-Längenfunktion gemessen
        add_start_index=True,                                   # fügt den Startindex des Chunks als Metadaten hinzu
    )

    chunks = text_splitter.split_documents(document_list)       # wir teilen die Dokumente in Chunks auf, wobei jedes Dokument eine Seite repräsentiert
    
    print(f"✂️ Das Dokument wurde in {len(chunks)} Chunks unterteilt.")     # Gibt die Anzahl der Chunks aus
    return chunks                                                          # Gibt die Liste der Chunks zurück


'''
Erstellt eine neue Vektordatenbank aus den Chunks.
'''

def create_vectordb(chunks, db_path):
   
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