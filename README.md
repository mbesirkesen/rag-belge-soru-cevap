# RAG Belge Soru-Cevap Sistemi

PDF ve TXT dosyalarından soru-cevap yapabilen Retrieval Augmented Generation (RAG) uygulaması. LangChain, ChromaDB ve Hugging Face modelleri kullanılarak geliştirilmiştir.

## Özellikler

- 📄 **PDF ve TXT Desteği**: PDF ve metin dosyalarını yükleyip indeksleyebilirsiniz
- 🔍 **Akıllı Arama**: ChromaDB vektör veritabanı ile semantik arama
- 💬 **Türkçe Soru-Cevap**: Türkçe sorulara Türkçe cevaplar
- 📚 **Kaynak Gösterimi**: Her cevabın hangi belgeden alındığını gösterir
- 🚫 **İlgisiz Soru Filtreleme**: Belgede olmayan bilgiler için "Bu belgeden çıkaramıyorum" yanıtı

## Kurulum

### Gereksinimler

- Python 3.11 veya 3.12 (Python 3.13 önerilmez - tokenizers uyumluluk sorunları olabilir)
- pip

### Adımlar

1. **Repoyu klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADI/REPO_ADI.git
cd REPO_ADI
```

2. **Sanal ortam oluşturun:**
```bash
python -m venv .venv
```

3. **Sanal ortamı etkinleştirin:**
   - Windows (PowerShell):
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   - Windows (CMD):
   ```cmd
   .venv\Scripts\activate.bat
   ```
   - Linux/Mac:
   ```bash
   source .venv/bin/activate
   ```

4. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

## Kullanım

### 1. İlk İndeks Oluşturma

`data/` klasörüne PDF veya TXT dosyalarınızı koyun, ardından:

```bash
python index_build.py
```

Bu komut `data/` klasöründeki tüm PDF ve TXT dosyalarını okuyup ChromaDB indeksini oluşturur.

### 2. Web Arayüzünü Başlatma

```bash
python app.py
```

Tarayıcınızda `http://127.0.0.1:7861` adresine gidin.

### 3. Dosya Yükleme ve Soru Sorma

1. Web arayüzünden PDF veya TXT dosyalarınızı yükleyin
2. "İndeksi Güncelle" butonuna tıklayın
3. Sorunuzu yazın ve Enter'a basın

## Proje Yapısı

```
.
├── app.py              # Gradio web arayüzü
├── core.py             # RAG mantığı (doküman yükleme, indeksleme, LLM)
├── index_build.py      # İlk indeks oluşturma scripti
├── requirements.txt    # Python bağımlılıkları
├── data/               # Yüklenen PDF/TXT dosyaları (gitignore'da)
└── chroma_db/          # ChromaDB vektör veritabanı (gitignore'da)
```

## Teknik Detaylar

### Kullanılan Teknolojiler

- **LangChain**: Doküman yükleme, bölme ve LLM entegrasyonu
- **ChromaDB**: Vektör veritabanı (embedding'ler için)
- **Hugging Face**: 
  - `sentence-transformers/all-MiniLM-L6-v2`: Embedding modeli
  - `google/flan-t5-small`: Metin üretim modeli
- **Gradio**: Web arayüzü
- **PDFMiner**: PDF metin çıkarımı

### Model Parametreleri

LLM için kullanılan parametreler (`core.py` içinde):
- `temperature=0.3`: Daha tutarlı cevaplar için düşük sıcaklık
- `repetition_penalty=1.4`: Tekrar azaltma
- `max_new_tokens=256`: Maksimum çıktı uzunluğu
- `num_beams=4`: Beam search ile daha iyi kalite

## Sorun Giderme

### "Token indices sequence length is longer" Uyarısı

Bu uyarı normaldir. `safe_compose_context()` fonksiyonu bağlamı otomatik olarak kısaltır.

### ChromaDB Hataları

Eğer indeks bozulursa, `chroma_db/` klasörünü silip `index_build.py` ile yeniden oluşturun.

### PDF Metin Çıkarımı Sorunları

Bazı PDF'lerde metin düzgün çıkarılamayabilir. Bu durumda PDF'in metin tabanlı (scan edilmiş değil) olduğundan emin olun.

## Lisans

Bu proje açık kaynaklıdır. Kendi kullanımınız için özgürce kullanabilirsiniz.

## Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır. Büyük değişiklikler için önce bir issue açarak neyi değiştirmek istediğinizi tartışın.

## İletişim

Sorularınız için issue açabilirsiniz.

