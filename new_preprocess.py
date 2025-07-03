import os
import json
from datetime import datetime
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from config import EMBEDDING_MODEL_NAME, DB_BASE_PATH


class OptimizedRAGPreprocessor:
    """
    Optimierter RAG-Preprocessor mit Docling HybridChunker
    """
    
    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 50):    # wir bauen den PDF-Reader mit diesen Einstellungen
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 1. Tokenizer erstellen - zählt die Tokens in einem Text
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
            max_tokens=max_tokens
        )
        
        # 2. Chunker erstellen
        # HybridChunker nutzt 2 Strategien gleichzeitig:
        # Struktur-bewusst: Achtet auf Kapitel, Absätze (wie Docling sie erkannt hat)
        # Token-bewusst: Achtet darauf, dass Chunks nicht zu lang werden
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True  # Kombiniert ähnliche benachbarte Chunks, wenn sie nicht zu lang sind
        )
        
        
        print(f"✅ OptimizedRAGPreprocessor initialisiert")
        print(f"📏 Max Tokens pro Chunk: {max_tokens}")
        print(f"🔄 Overlap Tokens: {overlap_tokens}")

    
    # LangChain Document (für Chroma) und Original Docling Document (für weitere Verarbeitung)
    def load_document_with_docling(self, file_path: str) -> tuple:
        """
        Lädt ein PDF mit Docling und gibt sowohl Documents als auch das Docling-Dokument zurück
        """
        if not os.path.exists(file_path):
            print(f"❌ Datei nicht gefunden: {file_path}")
            return [], None
        
        print(f"📖 Lade Dokument: {file_path}")
        
        # Docling Conversion
        converter = DocumentConverter()
        result = converter.convert(file_path)
        docling_doc = result.document
        
        # Vollständiges Markdown extrahieren
        full_markdown = docling_doc.export_to_markdown()
        
        # Dokument-Metadaten
        doc_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Ein Document für das gesamte PDF erstellen (für Docling Chunker)
        document = Document(
            page_content=full_markdown,
            metadata={
                'document_name': doc_name,
                'source': file_path,
                'extraction_method': 'docling',
                'total_chars': len(full_markdown)
            }
        )
        
        print(f"✅ Dokument geladen: {len(full_markdown)} Zeichen")
        
        return [document], docling_doc

    def create_chunks_with_docling(self, docling_doc, doc_metadata: Dict[str, Any]) -> List[Document]:
        """
        Erstellt Chunks mit dem Docling HybridChunker.
        Bekommt das von Docling geladene PDF-Dokument und allgemeine Infos zum Dokument (als Dictionary)
        """
        print("🧠 Erstelle Chunks mit Docling HybridChunker...")
        
        # Text zerteilen
        chunk_iter = self.chunker.chunk(dl_doc=docling_doc)     # ruft den HybridChunker auf --> Iterator mit Chunks
        docling_chunks = list(chunk_iter)                       # Der Iterator wird in eine echte Liste umgewandelt, damit man alle Chunks direkt verarbeiten kann
        
        print(f"📊 Docling HybridChunker ergab {len(docling_chunks)} Chunks")
        
        # Erstellt eine leere Liste, in die später die fertigen Chunks eingefügt werden
        langchain_chunks = []
        
        # iteriere über alle Docling Chunks
        for i, chunk in enumerate(docling_chunks):
            # Erzeugt neues Dictionary mit Metadaten für einen Chunk
            chunk_metadata = {
                **doc_metadata,
                'chunk_id': i,
                'chunk_length': len(chunk.text),
                'chunking_method': 'docling_hybrid'
            }
            
            # Prüft, ob der Chunk zusätzliche Metadaten (meta) besitzt
            if hasattr(chunk, 'meta') and chunk.meta:
                if hasattr(chunk.meta, 'headings') and chunk.meta.headings:        # Wenn der Chunk Überschriften enthält …
                    if isinstance(chunk.meta.headings, list):                   # … dann wird eine Liste der Überschriften zu einem String kombiniert (für Kompatibilität mit z. B. Chroma)
                        chunk_metadata['headings'] = ' | '.join(chunk.meta.headings)
                    else:
                        chunk_metadata['headings'] = str(chunk.meta.headings)
                
                if hasattr(chunk.meta, 'origin') and chunk.meta.origin:
                    chunk_metadata['origin_filename'] = str(chunk.meta.origin.filename)
                
                # Seitenzahlen extrahieren und als String speichern
                if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                    page_numbers = set()
                    for item in chunk.meta.doc_items:
                        if hasattr(item, 'prov') and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, 'page_no'):
                                    page_numbers.add(prov.page_no)  # Falls eine Seitenzahl (page_no) da ist → speichern
                    if page_numbers:
                        # Konvertiere Seitenzahlen zu String für Chroma-Kompatibilität
                        pages_sorted = sorted(list(page_numbers))
                        if len(pages_sorted) == 1:
                            chunk_metadata['page_numbers'] = str(pages_sorted[0])
                        else:
                            chunk_metadata['page_numbers'] = ','.join(map(str, pages_sorted))

        # Chroma speichert Metadaten als JSON, deshalb alles in Strings
            
            # Der Chunk wird als LangChain-Document gespeichert, mit Text und allen Metadaten, und zur Liste hinzugefügt
            langchain_chunks.append(Document(
                page_content=chunk.text,
                metadata=chunk_metadata
            ))
        
        # Chunk-Statistiken
        chunk_sizes = [len(chunk.page_content) for chunk in langchain_chunks]   # Liste aller Chunk-Längen (in Zeichen) erstellen
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            print(f"📏 Durchschnittliche Chunk-Größe: {avg_size:.0f} Zeichen")
            print(f"📏 Größter Chunk: {max(chunk_sizes)} Zeichen")
            print(f"📏 Kleinster Chunk: {min(chunk_sizes)} Zeichen")
        
        return langchain_chunks         # Gibt eine Liste von LangChain-Dokumenten, also Chunks zurück

    def save_markdown_export(self, docling_doc, doc_name: str) -> str:
        """
        Speichert den vollständigen Markdown-Export
        """
        # Markdown-Ordner erstellen
        markdown_dir = "markdown_exports"
        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir)
            print(f"📁 Ordner '{markdown_dir}' erstellt")
        
        # Markdown exportieren und speichern
        markdown_content = docling_doc.export_to_markdown()
        markdown_file = os.path.join(markdown_dir, f"{doc_name}.md")
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(f"# {doc_name}\n\n")
            f.write(f"*Exportiert am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"---\n\n")
            f.write(markdown_content)
        
        print(f"📝 Markdown gespeichert: {markdown_file}")
        return markdown_file

    def save_chunks_analysis(self, chunks: List[Document], doc_name: str) -> str:
        """
        Speichert detaillierte Chunk-Analyse
        """
        # Chunk-Analyse-Ordner erstellen
        analysis_dir = "chunk_analysis"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        
        # JSON-Export der Chunks (strukturiert)
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunks_data.append({
                'chunk_id': i,
                'content': chunk.page_content,
                'metadata': chunk.metadata,
                'length': len(chunk.page_content)
            })
        
        # JSON speichern
        json_file = os.path.join(analysis_dir, f"{doc_name}_chunks.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # Textdatei für menschenlesbare Analyse
        txt_file = os.path.join(analysis_dir, f"{doc_name}_chunks.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"CHUNK ANALYSIS: {doc_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Anzahl Chunks: {len(chunks)}\n")
            f.write(f"Chunking-Methode: Docling HybridChunker\n")
            f.write(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1:03d}\n")
                f.write(f"Länge: {len(chunk.page_content)} Zeichen\n")
                f.write(f"Metadata: {chunk.metadata}\n")
                f.write(f"Inhalt:\n{chunk.page_content}\n")
                f.write(f"{'='*60}\n\n")
        
        print(f"💾 Chunks gespeichert:")
        print(f"  📄 JSON: {json_file}")
        print(f"  📄 Text: {txt_file}")
        
        return analysis_dir

    def load_all_pdfs_in_folder(self) -> tuple:
        """
        Lädt alle PDFs im aktuellen Ordner.
        Gibt alle Chunks (für die Datenbank) und Infos über die Dateien aus
        """
        pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            print("❌ Keine PDF-Dateien im aktuellen Ordner gefunden.")
            return [], {}
        
        print(f"📚 {len(pdf_files)} PDF-Dateien gefunden:")
        for pdf in pdf_files:
            print(f"  - {pdf}")
        
        all_chunks = []
        all_exports = {}
        
        for pdf_file in pdf_files:
            print(f"\n📖 Verarbeite: {pdf_file}")
            
            # Dokument laden
            documents, docling_doc = self.load_document_with_docling(pdf_file)
            
            if not documents or not docling_doc:
                print(f"❌ Fehler beim Laden von {pdf_file}")
                continue
            
            doc_name = os.path.splitext(pdf_file)[0]
            
            # Markdown exportieren
            markdown_file = self.save_markdown_export(docling_doc, doc_name)
            
            # Chunks erstellen
            chunks = self.create_chunks_with_docling(
                docling_doc, 
                documents[0].metadata
            )
            
            # Chunk-Analyse speichern
            analysis_dir = self.save_chunks_analysis(chunks, doc_name)
            
            all_chunks.extend(chunks)
            all_exports[doc_name] = {
                'markdown_file': markdown_file,
                'chunks_count': len(chunks),
                'analysis_dir': analysis_dir
            }
        
        print(f"\n✅ Insgesamt {len(all_chunks)} Chunks aus {len(pdf_files)} PDFs erstellt")
        return all_chunks, all_exports

    def create_vectordb(self, chunks: List[Document], db_name: str = "optimized_rag") -> Chroma:
        """
        Erstellt Chroma-Vektordatenbank mit Metadaten-Filterung.
        Bekommt die Chunks, also Liste von Document-Objekten und optionalen Namen für die Datenbank.
        Gibt eine Chroma-Datenbank zurück.
        """
        db_path = os.path.join(DB_BASE_PATH, db_name)   # Erstelle vollständigen Pfad für die Vektordatenbank
        
        print(f"🛠️ Erstelle Vektordatenbank: {db_path}")
        print(f"📊 {len(chunks)} Chunks werden eingebettet...")
        
        # Metadaten bereinigen für Chroma-Kompatibilität
        cleaned_chunks = []
        for chunk in chunks:            # Für jeden Chunk...    
            cleaned_metadata = {}       # ... wird ein neues leeres Metadaten-Wörterbuch erstellt
            for key, value in chunk.metadata.items():
                if value is None:
                    cleaned_metadata[key] = ""  # Falls Wert None --> leerer String
                elif isinstance(value, (list, dict, tuple)):
                    cleaned_metadata[key] = str(value)      # Listen, Dict und Tupel --> Strings für Chroma
                elif isinstance(value, (str, int, float, bool)):
                    cleaned_metadata[key] = value           # diese Datentypen werden einfach übernommen
                else:
                    # Fallback: Konvertiere zu String
                    cleaned_metadata[key] = str(value)      # Alle anderen Datentypen → Fallback zu String
                                                            # alles ist JSON-kompatibel
            
            # Erstelle neues Document mit bereinigten Metadaten
            cleaned_chunks.append(Document(
                page_content=chunk.page_content,
                metadata=cleaned_metadata
            ))
        
        print(f"🔧 Metadaten für Chroma-Kompatibilität bereinigt")
        
        # Erstelle aus den cleaned_chunks neue Vektor-DB
        vectordb = Chroma.from_documents(
            documents=cleaned_chunks,
            embedding=self.embedding_model,     # wandelt Text in Embeddings um
            persist_directory=db_path           # Speicherort
        )
        vectordb.persist()                      # Datenbank dauerhaft auf der Festplatte gespeichert
        
        print(f"✅ Vektordatenbank gespeichert: {db_path}")
        return vectordb

    def process_all_pdfs(self, db_name: str = "optimized_rag_docling"):
        """
        Verarbeitet alle PDFs im Ordner und erstellt daraus eine Chroma-DB.
        """
        print("🔄 Optimiertes RAG-Preprocessing mit Docling startet...")
        
        db_path = os.path.join(DB_BASE_PATH, db_name)
        
        # Prüfen ob Datenbank bereits existiert
        if os.path.exists(db_path):
            print(f"🗃️ Datenbank existiert bereits: {db_path}")            
            vectordb = Chroma(
                persist_directory=db_path, 
                embedding_function=self.embedding_model
            )
            print(f"✅ Bestehende Datenbank geladen!")
            return vectordb
        
        print("🔄 Neue Datenbank wird erstellt...")
        
        # Lädt alle PDFs --> erstellt Chunks und sammelt Zusatzinfos (Anzahl Chunks pro Dokument etc.)
        all_chunks, exports_info = self.load_all_pdfs_in_folder()
        
        # Wenn keine Chunks erstellt wurden (z. B. weil keine PDFs) --> Abbruch
        if not all_chunks:
            print("❌ Keine Chunks erstellt.")
            return None
        
        # Vektordatenbank erstellen
        vectordb = self.create_vectordb(all_chunks, db_name)
        
        # Zusammenfassung
        print(f"\n🎉 Verarbeitung abgeschlossen!")
        print(f"📁 Markdown-Exports: markdown_exports/")
        print(f"📁 Chunk-Analysen: chunk_analysis/")
        print(f"🗃️ Vektordatenbank: {db_path}")
        print(f"📊 Gesamt-Chunks: {len(all_chunks)}")
        
        # Export-Details
        print(f"\n📋 Verarbeitete Dokumente:")
        for doc_name, info in exports_info.items():
            print(f"  📄 {doc_name}: {info['chunks_count']} Chunks")
        
        return vectordb


def main():
    """
    Hauptfunktion für das optimierte Preprocessing
    """
    # Initialisiere den Preprocessor
    preprocessor = OptimizedRAGPreprocessor(
        max_tokens=400,  # Anpassbar je nach Bedarf
        overlap_tokens=50
    )
    
    # Verarbeite alle PDFs
    vectordb = preprocessor.process_all_pdfs(db_name="optimized_rag_docling")
    
    if vectordb:
        print("\n🔍 Teste die Datenbank mit einer Beispielsuche...")
        # Beispielsuche
        results = vectordb.similarity_search("Was ist das Hauptthema?", k=10)
        print(f"🎯 {len(results)} Ergebnisse gefunden!")
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Ergebnis {i}:")
            print(f"Text (erste 150 Zeichen): {result.page_content[:150]}...")
            print(f"Metadata: {result.metadata}")


if __name__ == "__main__":
    main()