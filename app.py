import os
import sys
import traceback
import gradio as gr
from dotenv import load_dotenv

# .env yükle
load_dotenv()

# core modülünden importlar
from core import (
    DATA_DIR,
    PERSIST_DIR,
    ensure_dirs,
    build_or_update_vectorstore,
    load_documents,
    split_documents,
    get_retriever,
    get_llm,
    safe_compose_context,
    context_is_relevant,
    reduce_repetition,
    build_prompt,
    format_sources_by_document,
    fetch_weather_summary,
)


def clean_llm_output(text: str) -> str:
    """
    LLM çıktısını temizler: Prompt talimatlarını, kural açıklamalarını vs. kaldırır.
    """
    import re
    
    if not text or len(text) < 5:
        return text
    
    # Prompt talimatlarını ve kuralları kaldır
    lines = text.split('\n')
    cleaned_lines = []
    
    skip_patterns = [
        r'KRİTİK KURALLAR',
        r'ÖNEMLİ:',
        r'Bağlamda.*?çıkaramıyorum',
        r'Sadece bağlamda.*?kullan',
        r'varsayım yapma',
        r'Örnek:',
        r'^\d+\.\s+',  # 1. 2. gibi numaralı liste
    ]
    
    for line in lines:
        line_lower = line.lower()
        should_skip = False
        
        for pattern in skip_patterns:
            if re.search(pattern, line_lower):
                should_skip = True
                break
        
        if not should_skip and line.strip():
            cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    
    # Fazla boşlukları temizle
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Eğer çok kısa kaldıysa orijinal metni döndür
    if len(cleaned) < 10 and len(text) > 50:
        return text.strip()
    
    return cleaned if cleaned else text.strip()


def compose_memory_text(history_state: list[tuple[str, str]]) -> str:
    """
    Önceki konuşmaları kısa bir hafıza metnine çevirir.
    """
    if not history_state:
        return ""

    entries = history_state[-6:]
    lines = []
    for user_msg, bot_msg in entries:
        lines.append(f"Kullanıcı: {user_msg}")
        lines.append(f"Bot: {bot_msg}")

    return "\n".join(lines)


def reset_data_store() -> str:
    import shutil

    paths = [PERSIST_DIR, DATA_DIR]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    ensure_dirs()
    return "Veri klasörleri temizlendi (chroma_db + data)."


def ui_answer(
    message,
    history_state,
    k_value,
    use_mmr,
    prompt_format,
    use_turkish_embedding,
    weather_city,
    use_weather_api,
):
    """
    Sohbet Fonksiyonu
    Input: Mesaj ve Gizli Geçmiş (history_state)
    Output: (Boş mesaj kutusu, Chatbot görünümü, Güncel Gizli Geçmiş)
    """
    if history_state is None:
        history_state = []

    try:
        # Query expansion: Eğitim soruları için daha spesifik kelimeler ekle
        expanded_query = message
        query_lower = message.lower()
        # Eğitim soruları için CV'nin eğitim bölümünü bulmak için daha spesifik kelimeler
        if any(kw in query_lower for kw in ["eğitim", "okul", "üniversite", "mezun", "bölüm", "cv", "özgeçmiş"]):
            # Eğitim sorularında sadece eğitimle ilgili kelimeler kullan, proje kelimelerini kullanma
            expanded_query = message + " üniversite okul mezun bölüm fakülte eğitim öğrenci lisans yüksek"
        elif any(kw in query_lower for kw in ["beceri", "yetenek", "programlama", "dil"]):
            expanded_query = message + " beceri yetenek programlama dil teknoloji araç"
        
        # RAG İşlemleri
        retriever = get_retriever(
            PERSIST_DIR, 
            k=int(k_value), 
            use_mmr=use_mmr,
            turkish_focused=use_turkish_embedding
        )
        docs = retriever.invoke(expanded_query)
        context = safe_compose_context(docs)

        # İsteğe bağlı: Hava durumu API özeti
        api_summary = ""
        if use_weather_api:
            try:
                api_text, is_real = fetch_weather_summary(weather_city)
                api_summary = api_text
            except Exception as api_err:
                print(f"Hava durumu API hatası (UI): {api_err}")
        
        # Minimal debug (sadece hata durumlarında)
        if not context or not context_is_relevant(message, context):
            print(f"⚠️ Soru: '{message[:50]}...' - Context yetersiz veya ilgisiz")

        # Memory metnini hazırla
        memory_text = compose_memory_text(history_state)

        # Belge bağlamı + hafıza + API özeti tek promptta birleşsin
        parts = []
        if memory_text:
            parts.append(f"Önceki konuşmalar:\n{memory_text}")
        if context:
            parts.append(f"Belge bağlamı:\n{context}")
        if api_summary:
            parts.append(f"API (hava durumu) özeti:\n{api_summary}")

        full_context = "\n\n".join(parts).strip()

        llm = get_llm()
        prompt = build_prompt(message, full_context, prompt_format, sources=docs)
        
        if not full_context or not context_is_relevant(message, full_context):
            answer = "Bu belgeden çıkaramıyorum."
            sources_text = ""
        else:
            answer = llm.invoke(prompt).strip()
            answer = reduce_repetition(answer)
            
            # LLM çıktısını temizle: Prompt talimatlarını, kural açıklamalarını vs. kaldır
            answer = clean_llm_output(answer)
            
            # Kaynakları belge bazında formatla
            sources_text = format_sources_by_document(docs)

        # Kaynakları ekle (belge bazında gruplanmış)
        if sources_text:
            full_response = answer + f"\n\n📚 Kaynaklar:\n{sources_text}"
        else:
            full_response = answer

        # Geçmişe ekle (User, Bot)
        history_state.append((message, full_response))
        
        # 1. Msg kutusunu temizle ("")
        # 2. Chatbot'a geçmişi ver (history_state)
        # 3. State'i güncelle (history_state)
        return "", history_state, history_state

    except Exception as e:
        print(f"Hata: {e}")
        traceback.print_exc()
        error_msg = f"Hata oluştu: {str(e)}"
        history_state.append((message, error_msg))
        return "", history_state, history_state


def ui_upload(files):
    """
    Dosya yükleme fonksiyonu
    """
    print("--- Dosya Yükleme İsteği Geldi ---")
    
    # 422 Hatası için güvenlik kontrolü
    if files is None:
        return "⚠️ Dosya seçilmedi."
        
    # Gradio versiyonuna göre files listesi veya tek obje gelebilir
    file_paths = []
    
    # Liste mi değil mi kontrol et
    if isinstance(files, list):
        for f in files:
            # Gradio dosyayı bir nesne olarak mı yoksa string yol olarak mı gönderiyor?
            if isinstance(f, str):
                file_paths.append(f)
            elif hasattr(f, 'name'): # Gradio temp file objesi
                file_paths.append(f.name)
    elif isinstance(files, str):
        file_paths.append(files)
    elif hasattr(files, 'name'):
        file_paths.append(files.name)

    if not file_paths:
        return "⚠️ Dosya yolu okunamadı."

    print(f"İşlenecek dosyalar: {file_paths}")
    
    saved_paths = []
    ensure_dirs()

    try:
        # Dosyaları kopyala
        import shutil
        for src_path in file_paths:
            filename = os.path.basename(src_path)
            target_path = os.path.join(DATA_DIR, filename)
            shutil.copy2(src_path, target_path)
            saved_paths.append(target_path)

        # İndeksle
        docs = load_documents(saved_paths)
        if not docs:
            return "❌ Metin okunamadı."
            
        chunks = split_documents(docs)
        build_or_update_vectorstore(chunks, persist_directory=PERSIST_DIR)
        
        return f"✅ Başarılı! {len(chunks)} parça eklendi."

    except Exception as e:
        print(f"Upload Hatası: {e}")
        traceback.print_exc()
        return f"❌ Hata: {str(e)}"


def build_demo():
    ensure_dirs()
    
    # CSS: Chatbot yüksekliği
    css = "#chatbot {height: 500px !important; overflow: auto;}"

    with gr.Blocks(title="RAG Asistanı", css=css) as demo:
        # --- STATE (Gizli Hafıza) ---
        # Chatbot'u input olarak kullanmak yerine bunu kullanacağız
        history_state = gr.State([]) 

        gr.Markdown("## 📄 Belge Soru-Cevap Sistemi")

        with gr.Row():
            with gr.Column(scale=1):
                file_uploader = gr.File(
                    label="PDF/TXT Yükle", 
                    file_types=[".pdf", ".txt"], 
                    file_count="multiple",
                    type="filepath" # Bu ayar önemli
                )
                upload_btn = gr.Button("İndeksi Güncelle", variant="primary")
                upload_info = gr.Textbox(label="Durum", interactive=False)
                data_reset_status = gr.Textbox(label="Veri Durumu", interactive=False)
                clear_data_btn = gr.Button("Veri Temizle (chroma_db + data)", variant="secondary")
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Sohbet", elem_id="chatbot")
                with gr.Row():
                    msg = gr.Textbox(
                        label="Sorunuz", 
                        placeholder="Yazın ve Enter'a basın...",
                        scale=4
                    )
                    clear = gr.Button("Temizle", scale=1)

        with gr.Accordion("Ayarlar", open=False):
            k_slider = gr.Slider(
                minimum=1,
                maximum=15,
                value=6,
                step=1,
                label="k (CV soruları için 8-10 önerilir)",
            )
            mmr_checkbox = gr.Checkbox(value=True, label="MMR")
            prompt_format = gr.Dropdown(
                ["kısa", "madde", "özet_madde", "önce_sonuç"],
                value="kısa",
                label="Format",
            )
            turkish_embedding = gr.Checkbox(
                value=False,
                label="TR Embedding",
            )

            gr.Markdown("### 🌤 Hava Durumu API Ayarları")
            weather_city = gr.Textbox(
                label="Şehir (hava durumu için)",
                placeholder="Örn: İstanbul",
                value="İstanbul",
            )
            use_weather_api = gr.Checkbox(
                value=False,
                label="Hava durumu API'sini kullan (OPENWEATHER_API_KEY yoksa MOCK veri döner)",
            )

        # --- OLAYLAR (EVENTS) ---
        
        # 1. Upload Butonu
        upload_btn.click(
            fn=ui_upload, 
            inputs=[file_uploader], 
            outputs=[upload_info]
        )
        clear_data_btn.click(fn=reset_data_store, inputs=None, outputs=[data_reset_status])

        # 2. Mesaj Gönderme
        # DİKKAT: inputs içinde 'chatbot' YOK. 'history_state' VAR.
        msg.submit(
            fn=ui_answer,
            inputs=[
                msg,
                history_state,
                k_slider,
                mmr_checkbox,
                prompt_format,
                turkish_embedding,
                weather_city,
                use_weather_api,
            ],
            outputs=[msg, chatbot, history_state],  # Çıktı sırası: MesajKutusu, GörselChat, GizliState
        )

        # 3. Temizle
        def clear_history():
            return [], [] # Hem chatbot hem state temizlenir
            
        clear.click(fn=clear_history, inputs=None, outputs=[chatbot, history_state])

    return demo

def _read_server_port() -> int:
    raw = os.getenv("SERVER_PORT", "7860")
    try:
        return int(raw)
    except ValueError:
        print(f"⚠️ SERVER_PORT değeri geçersiz: {raw}. Varsayılan 7860 kullanılacak.")
        return 7860


if __name__ == "__main__":
    base_port = _read_server_port()
    
    demo = build_demo()
    
    # Port çakışması durumunda alternatif portları dene
    for port_offset in range(10):
        try:
            port = base_port + port_offset
            print(f"Başlatılıyor... Port: {port}")
            demo.launch(server_port=port, share=False)
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) or "Port" in str(e):
                print(f"Port {port} kullanımda, {port + 1} deniyor...")
                continue
            else:
                raise
    else:
        print(f"❌ 7860-7869 arası tüm portlar kullanımda!")
        print("Lütfen bir Python işlemini durdurun veya SERVER_PORT değişkenini ayarlayın.")