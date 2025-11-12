import os
from typing import List, Tuple, Union

import gradio as gr
from langchain_core.documents import Document

from core import (
    DATA_DIR,
    PERSIST_DIR,
    ensure_dirs,
    build_or_update_vectorstore,
    load_documents,
    split_documents,
    get_retriever,
    format_source_list,
    get_llm,
    safe_compose_context,
    context_is_relevant,
    reduce_repetition,
)


def _extract_path(item: Union[str, dict, object]) -> Union[str, None]:
    # Gradio Files bileşeni farklı şekillerde değer döndürebilir.
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("name") or item.get("path")
    # tempfile benzeri nesneler
    name = getattr(item, "name", None)
    return name if isinstance(name, str) else None


def save_uploaded_files(files: List[Union[str, dict, object]]) -> List[str]:
    ensure_dirs()
    saved_paths: List[str] = []
    # Gradio bazı sürümlerde [ [fileDict, ...] ] şeklinde iç içe liste döndürebilir
    if files and isinstance(files, list) and files and isinstance(files[0], list):
        files = files[0]  # type: ignore[assignment]
    # Tek dosya dict'i gelebilir
    if files and isinstance(files, dict):  # type: ignore[redundant-cast]
        files = [files]  # type: ignore[assignment]
    for item in files or []:
        src_path = _extract_path(item)
        if not src_path or not os.path.exists(src_path):
            continue
        filename = os.path.basename(src_path)
        target = os.path.join(DATA_DIR, filename)
        with open(src_path, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
        saved_paths.append(target)
    return saved_paths


def rebuild_index_from_saved(paths: List[str]) -> int:
    docs = load_documents(paths)
    chunks = split_documents(docs)
    build_or_update_vectorstore(chunks, persist_directory=PERSIST_DIR)
    return len(chunks)


def ui_answer(message, history):
    retriever = get_retriever(PERSIST_DIR)
    docs = retriever.invoke(message)
    context = safe_compose_context(docs)
    llm = get_llm()
    prompt = (
        "Aşağıdaki bağlamı kullanarak soruya Türkçe ve kısa yanıt ver. "
        "Bağlamda açık bilgi yoksa 'Bu belgeden çıkaramıyorum.' de.\n\n"
        f"Soru: {message}\n\nBağlam:\n{context}"
    )
    if not context or not context_is_relevant(message, context):
        answer = "Bu belgeden çıkaramıyorum."
    else:
        try:
            answer = llm.invoke(prompt).strip()
            answer = reduce_repetition(answer)
        except Exception as exc:  # token aşımı vb. durumlar
            answer = f"Hata oluştu: {exc}"
    source_list = [d.metadata.get("source", "?") for d in docs]
    # tekrarları kaldır
    source_list = list(dict.fromkeys(source_list))
    # kaynakları biçimlendir
    sources = [(s, "") for s in source_list]
    source_text = "\n".join(f"- {src} {loc}".strip() for src, loc in sources)
    answer_with_src = answer + ("\n\nKaynaklar:\n" + source_text if sources else "")
    history = history + [(message, answer_with_src)]
    return history, history


def ui_upload(files=None):
    saved = save_uploaded_files(files)
    if not saved:
        return gr.update(value="Yükleme yapılmadı."), gr.update(value=[])
    chunks = rebuild_index_from_saved(saved)
    return gr.update(value=f"Yüklendi: {len(saved)} dosya. İndekse eklenen parça: {chunks}"), gr.update(value=[])


def build_demo():
    ensure_dirs()
    with gr.Blocks(title="Belge Soru-Cevap (LangChain + Chroma)") as demo:
        gr.Markdown("**PDF/TXT yükle, indeksle ve sorular sor.**")

        with gr.Row():
            # Gradio 3.34 için daha stabil: tek bileşenle çoklu seçim
            file_uploader = gr.File(label="PDF/TXT yükle", file_types=[".pdf", ".txt"], file_count="multiple")
        upload_info = gr.Textbox(label="Durum", interactive=False)
        upload_btn = gr.Button("İndeksi Güncelle")

        chatbot = gr.Chatbot(label="Asistan")
        msg = gr.Textbox(label="Sorunuz", placeholder="Soruyu yazın ve Enter'a basın...")
        clear = gr.Button("Temizle")

        # Dosya seçildiğinde otomatik indeksleme (bazı sürümlerde buton+Files 422 üretebiliyor)
        file_uploader.change(ui_upload, inputs=[file_uploader], outputs=[upload_info, chatbot])
        upload_btn.click(ui_upload, inputs=[file_uploader], outputs=[upload_info, chatbot])
        msg.submit(ui_answer, inputs=[msg, chatbot], outputs=[chatbot, chatbot])
        clear.click(lambda: ([], []), outputs=[chatbot, chatbot])

        gr.Markdown(
            "Varsayılan indeks klasörü: `chroma_db`. Veri klasörü: `data`."
            " Cevapların sonunda kullanılan kaynak dosyalar listelenir."
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_port=7861,
        server_name="127.0.0.1",
        share=False,
        inbrowser=False
    )

