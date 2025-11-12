from pathlib import Path
from typing import List

from langchain_community.document_loaders import PDFMinerLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Klasör ayarları
DATA_DIR = Path("data")
DB_DIR = "chroma_db"


def load_docs(data_dir: Path) -> List:
    docs = []
    for p in data_dir.glob("**/*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            loader = PDFMinerLoader(str(p))
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = str(p.relative_to(Path.cwd()))
            docs += loaded
        elif suffix in {".txt", ".md"}:
            docs += TextLoader(str(p), encoding="utf-8").load()
    return docs


def build_index() -> int:
    print("Belgeler yükleniyor...")
    raw_docs = load_docs(DATA_DIR)
    if not raw_docs:
        print(f"[{DATA_DIR}] dizininde doküman/sayfa bulunamadı.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"{len(raw_docs)} doküman/sayfa bulundu.")
    print(f"{len(chunks)} parça (chunk) oluşturuldu.")

    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeds, persist_directory=DB_DIR)
    # langchain-chroma 0.1+: otomatik kalıcı; persist() metodu yok

    print(f"Chroma indeks oluşturuldu: [{DB_DIR}/]")
    return len(chunks)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_index()
