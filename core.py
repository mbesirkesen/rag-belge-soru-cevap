import os
from typing import Iterable, List, Tuple

from langchain_community.document_loaders import PDFMinerLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_from_txt(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Keep file reference in metadata for source display
    return [Document(page_content=text, metadata={"source": os.path.relpath(path), "type": "txt"})]


def _load_from_pdf(path: str) -> List[Document]:
    """
    PDF metnini daha temiz çıkarmak için PDFMiner kullan.
    (PyPDF bazı dosyalarda CID/Unicode bozulmaları üretebiliyor)
    """
    loader = PDFMinerLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = os.path.relpath(path)
    return docs


def load_documents(paths: Iterable[str]) -> List[Document]:
    documents: List[Document] = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            documents.extend(_load_from_txt(path))
        elif ext == ".pdf":
            documents.extend(_load_from_pdf(path))
        else:
            continue
    return documents


def split_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def get_embeddings(turkish_focused: bool = False):
    """
    Embedding modeli döndürür.
    
    Args:
        turkish_focused: True ise Türkçe odaklı çok dilli model kullanır
    """
    if turkish_focused:
        # Türkçe için optimize edilmiş çok dilli model
        return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # Small, CPU-friendly model (varsayılan)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_or_update_vectorstore(chunks: List[Document], persist_directory: str = PERSIST_DIR, turkish_focused: bool = False) -> Chroma:
    os.makedirs(persist_directory, exist_ok=True)
    embeddings = get_embeddings(turkish_focused=turkish_focused)
    
    # If a store exists, add to it; otherwise create new
    # ChromaDB'nin varlığını kontrol etmek için daha güvenli yöntem
    db_exists = False
    try:
        existing_files = os.listdir(persist_directory)
        # ChromaDB en azından birkaç dosya oluşturur
        db_exists = len(existing_files) > 0 and any(f.endswith('.sqlite3') or f == 'chroma.sqlite3' for f in existing_files)
    except:
        db_exists = False
    
    if db_exists:
        try:
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            vectorstore.add_documents(chunks)
            # langchain-chroma 0.1+: otomatik kalıcı; persist() yok
            return vectorstore
        except Exception as e:
            print(f"Varolan indekse ekleme hatası: {e}. Yeni indeks oluşturuluyor...")
            # Hata varsa yeni oluştur
            return Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    else:
        return Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)


def get_retriever(persist_directory: str = PERSIST_DIR, k: int = 6, use_mmr: bool = True, turkish_focused: bool = False):
    """
    Retriever döndürür.
    
    Args:
        persist_directory: ChromaDB dizin yolu
        k: Döndürülecek doküman sayısı (3-6 arası önerilir)
        use_mmr: True ise Maximum Marginal Relevance kullanır (çeşitlilik için)
        turkish_focused: True ise Türkçe odaklı embedding modeli kullanır
    """
    embeddings = get_embeddings(turkish_focused=turkish_focused)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    if use_mmr:
        # MMR ile çeşitlilik
        fetch_k = max(k * 3, 20)  # fetch_k en az k*3 veya 20
        return vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5}
        )
    else:
        # Basit similarity search
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )


def discover_data_files(root_dir: str = DATA_DIR) -> List[str]:
    os.makedirs(root_dir, exist_ok=True)
    file_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                file_paths.append(os.path.join(dirpath, name))
    return file_paths


def format_source_list(docs: List[Document]) -> List[Tuple[str, str]]:
    """
    Returns list of tuples (source, location) e.g. (my.pdf, "page 3") or (a.txt, "chunk at 1234")
    """
    items: List[Tuple[str, str]] = []
    for d in docs:
        source = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        page = d.metadata.get("page")
        start_index = d.metadata.get("start_index")
        location = f"page {page}" if page is not None else f"offset {start_index}" if start_index is not None else ""
        items.append((source, location))
    return items


def format_sources_by_document(docs: List[Document]) -> str:
    """
    Kaynakları belge bazında gruplayarak formatlar.
    Her belge için sayfa bilgilerini toplar ve ayrı satırlarda gösterir.
    
    Returns:
        Formatlanmış kaynak string'i
    """
    from collections import defaultdict
    
    # Belge bazında grupla: {source: [pages]}
    doc_pages = defaultdict(set)
    for d in docs:
        source = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        # Sadece dosya adını al (tam yol yerine)
        filename = os.path.basename(source)
        
        page = d.metadata.get("page")
        if page is not None:
            doc_pages[filename].add(page)
        else:
            # Sayfa yoksa boş set ile işaretle
            if filename not in doc_pages:
                doc_pages[filename] = set()
    
    # Formatla: Her belge için sayfaları sırala
    lines = []
    for filename, pages in sorted(doc_pages.items()):
        if pages:
            sorted_pages = sorted(pages)
            if len(sorted_pages) == 1:
                lines.append(f"📄 {filename} (sayfa {sorted_pages[0]})")
            else:
                # Birden fazla sayfa varsa aralık göster (örn: sayfa 3-5, 7, 9)
                page_str = format_page_range(sorted_pages)
                lines.append(f"📄 {filename} ({page_str})")
        else:
            lines.append(f"📄 {filename}")
    
    return "\n".join(lines) if lines else ""


def format_page_range(pages: List[int]) -> str:
    """
    Sayfa listesini okunabilir formata çevirir.
    Örnek: [1, 2, 3, 5, 7, 8] -> "1-3, 5, 7-8"
    (Sadece sayfa numaralarını döndürür, "sayfa" kelimesi eklenmez)
    """
    if not pages:
        return ""
    
    pages = sorted(set(pages))
    if len(pages) == 1:
        return str(pages[0])
    
    ranges = []
    start = pages[0]
    end = pages[0]
    
    for i in range(1, len(pages)):
        if pages[i] == end + 1:
            end = pages[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = pages[i]
    
    # Son aralığı ekle
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    
    return ", ".join(ranges)


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)


# --- LLM yardımcıları (giriş uzunluğu güvenli sınırlar) ---
def get_llm() -> HuggingFacePipeline:
    """
    Küçük ve CPU-dostu T5 modeli (otomatik truncation/padding ile).
    """
    from transformers import pipeline as hf_pipeline

    text2text = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        truncation=True,
        max_new_tokens=256,
        num_beams=4,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.4,
        no_repeat_ngram_size=6,
        length_penalty=0.2,
        early_stopping=True,
    )
    return HuggingFacePipeline(pipeline=text2text)


def safe_compose_context(docs: List[Document], max_tokens: int = 1024) -> str:
    """
    Doküman içeriğini HF tokenizer userına gerek kalmadan kaba token tahminiyle sınırlar.
    """
    approx_ratio = 0.75  # Türkçe için 1 token ~ 0.75 kelime tahmini
    max_chars = int(max_tokens * approx_ratio * 4)  # ~4 karakter / kelime varsayımı

    chunks: List[str] = []
    total = 0
    for d in docs:
        part = (d.page_content or "")
        # metni normalize et
        part = " ".join(part.split())
        if not part:
            continue
        take = min(len(part), max(0, max_chars - total))
        if take <= 0:
            break
        chunks.append(part[:take])
        total += take
        if total >= max_chars:
            break
    return "\n\n".join(chunks)


def context_is_relevant(query: str, context: str) -> bool:
    """
    Uygunluk kontrolü: sorudan çıkan anahtar kelimelerin
    en azından bir kısmı bağlamda geçmeli; değilse reddet.
    CV ve kişisel bilgi soruları için daha esnek.
    """
    if not context or len(context.strip()) < 20:
        return False
    
    q = " ".join(query.lower().split())
    c = context.lower()
    
    # Türkçe stop sözcüklerin küçük bir alt kümesi
    stops = {"ve", "ile", "da", "de", "mi", "bir", "için", "ne", "mı", "mü", "ya", "ama", "veya", "nedir", "nelerdir", "misin", "misiniz", "özetler", "özet"}
    
    # CV/kişisel bilgi soruları için özel kelimeler
    personal_keywords = {"eğitim", "üniversite", "okul", "mezun", "bölüm", "bölümü", "ad", "isim", "adım", "kimim", "kim", 
                         "beceri", "yetenek", "proje", "deneyim", "iş", "çalışma", "sertifika", "dil", "iletişim", "telefon",
                         "cv", "özgeçmiş", "bilgi", "detay", "hakkında"}
    
    # Eğer soru kişisel bilgi içeriyorsa ve context varsa, çok esnek ol
    query_has_personal = any(kw in q for kw in personal_keywords)
    if query_has_personal:
        # CV soruları için çok toleranslı: context varsa genelde kabul et
        # Sadece çok kısa context'leri reddet
        if len(context.strip()) > 30:
            # İki kelime bile eşleşirse kabul et
            terms = [t for t in q.split() if t not in stops and len(t) > 2]
            if terms:
                hit = sum(1 for t in terms if t in c)
                if hit >= 1:
                    return True
            # Hiç kelime eşleşmese bile, context uzunsa kabul et (çok toleranslı)
            if len(context.strip()) > 100:
                return True
    
    # Normal kontrol (diğer sorular için)
    terms = [t for t in q.split() if t not in stops and len(t) > 3]
    if not terms:
        # Eğer soru çok kısa veya sadece stop kelimeler içeriyorsa, context varsa kabul et
        if len(context.strip()) > 50:
            return True
        return False
    hit = sum(1 for t in terms if t in c)
    return hit >= max(1, len(terms) // 3)


def reduce_repetition(text: str) -> str:
    """
    Basit tekrar azaltma: ardışık aynı kelimeleri ve 3-4 kelimelik
    tekrar eden ngramları sıkıştırır.
    """
    import re
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\b(\w+)(?:\s+\1){1,}\b", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(\b\w+(?:\s+\w+){2,3}\b)(?:\s+\1){1,}", r"\1", t, flags=re.IGNORECASE)
    return t


def build_prompt(query: str, context: str, prompt_format: str = "kısa", sources: List[Document] = None) -> str:
    """
    Prompt oluşturur.
    
    Args:
        query: Kullanıcı sorusu
        context: Belge bağlamı
        prompt_format: Prompt formatı seçeneği
            - "kısa": Kısa ve öz yanıt
            - "madde": Madde madde liste
            - "özet_madde": Önce 1 cümle özet, sonra 3 madde
            - "önce_sonuç": Önce sonuç, sonra gerekçe
        sources: Kaynak dokümanlar (her belgeden 1 cümle kuralı için)
    """
    base_instruction = "Bağlamı kullanarak soruya Türkçe yanıt ver. Bağlamda soruya doğrudan cevap verecek bilgi yoksa sadece 'Bu belgeden çıkaramıyorum.' yaz."
    
    if prompt_format == "kısa":
        instruction = base_instruction + " Kısa ve net cevap ver.\n\n"
    elif prompt_format == "madde":
        instruction = base_instruction + " Yanıtı madde madde ver.\n\n"
    elif prompt_format == "özet_madde":
        instruction = base_instruction + " Önce 1 cümle özet, sonra 3 madde halinde detaylandır.\n\n"
    elif prompt_format == "önce_sonuç":
        instruction = base_instruction + " Önce kısa sonuç (1-2 cümle), sonra gerekçesini açıkla.\n\n"
    else:
        instruction = base_instruction + "\n\n"
    
    # Her belgeden en az 1 cümle kuralı (çoklu kaynak varsa) - sadece özet_madde formatında
    if sources and prompt_format == "özet_madde":
        unique_sources = set()
        for doc in sources:
            source = doc.metadata.get("source") or "unknown"
            unique_sources.add(os.path.basename(source))
        
        if len(unique_sources) > 1:
            instruction += f"{len(unique_sources)} belgeden bilgi var, her birinden örnek ver.\n\n"
    
    return f"{instruction}Soru: {query}\n\nBağlam:\n{context}\n\nCevap:"

