# Qdrant Chatbot API Service

Implementasi EPIC 1–5: health check, query embedding, pencarian chunk, reranking
opsional, chat multi-turn, query rewriting, dan jawaban JSON/SSE dengan sitasi.
Service hanya membaca koleksi Qdrant yang sudah ada.

## Menjalankan

Gunakan Python 3.12+ dan [uv](https://docs.astral.sh/uv/). Dari folder `ai-service/`:

```powershell
uv sync --locked
uv run uvicorn main:app --reload
```

Swagger UI tersedia di `http://127.0.0.1:8000/docs`.
Jika `uv` belum tersedia di PATH tetapi sudah terinstal sebagai Python module,
gunakan `python -m uv` sebagai pengganti `uv`.

`core/config.py` membaca `.env` di folder service, terlepas dari working directory.
Environment variables memiliki prioritas atas `.env`. Gunakan file `.env` yang
sudah disiapkan; jangan memasukkan kredensial ke source control.

Konfigurasi koleksi: `plp_indoe5_cs96_ov32`, unnamed vector, dimensi 384, cosine,
payload teks `text`, model `LazarusNLP/all-indo-e5-small-v4`.
`QDRANT_VECTOR_NAME` kosong diperlakukan sebagai `None`.

Startup memvalidasi keberadaan koleksi, dimensi, nama vector, dan cosine distance.
Startup gagal jika konfigurasi tidak sesuai; service tidak membuat atau mengubah
koleksi. Satu `AsyncQdrantClient` digunakan selama lifespan aplikasi dan ditutup
saat shutdown.

## API

`GET /health` memeriksa Qdrant setiap dipanggil dan mengembalikan:

```json
{
  "status": "ok",
  "qdrant_status": "connected",
  "collection_name": "plp_indoe5_cs96_ov32",
  "points_count": 677,
  "vector_dimension": 384
}
```

Jumlah points berasal dari koleksi saat probe. Koneksi gagal menghasilkan HTTP 503,
status `degraded`, `qdrant_status=disconnected`, dan count/dimensi `null`.

`POST /api/v1/retrieve` menerima:

```json
{
  "query": "Bagaimana prosedur pelaksanaan PLP?",
  "top_k": 5,
  "score_threshold": 0.5,
  "filters": {},
  "use_rerank": false,
  "top_k_initial": 15
}
```

Respons berbentuk `{"query": "...", "results": [...], "total": 5}`. Setiap hasil
memiliki `point_id`, `score` cosine dari Qdrant, `text`, `metadata`, serta
`rerank_score` (`null` jika tidak direrank). Semua payload selain field teks
dipertahankan di metadata, termasuk `chunk_id`, `doc_index`, `page`,
`source_document`, `chapter`, `section`, dan `config_id` bila tersedia.
Point tanpa teks valid dilewati. Pencarian tanpa hasil mengembalikan list kosong.

Aktifkan `use_rerank=true` (alias `use_reranker` juga diterima) untuk mengambil
`top_k_initial` kandidat, kemudian mererank dengan FlashRank multilingual
`ms-marco-MultiBERT-L-12`. Hasil akhir dibatasi `top_k_final`, atau `top_k` jika
`top_k_final` tidak diberikan. `top_k_initial` harus minimal sebesar batas akhir.
`top_k_final` hanya digunakan bersama reranking. Skor asli Qdrant tetap disimpan
di `score`; threshold diterapkan sebelum reranking.

Filter dapat berupa pencocokan payload sederhana seperti `{"chapter": "I"}`,
list nilai seperti `{"page": [2, 3]}`, atau JSON filter Qdrant asli seperti:

```json
{"must": [{"key": "page", "range": {"gte": 2, "lte": 5}}]}
```

Filter mengikuti struktur payload asli; field yang tidak terindeks dapat ditolak
oleh Qdrant Cloud jika strict mode aktif. Parameter tidak valid menghasilkan
HTTP 422; kegagalan dependency retrieval menghasilkan HTTP 503.

Model embedding dan reranker dimuat saat pertama digunakan dan dapat mengunduh
file model saat cache belum tersedia. Inferensi berjalan di worker thread.
Embedding menggunakan `prompt=""` dan `normalize_embeddings=True`, lalu diperiksa
dimensi dan normanya sebelum query Qdrant.

## Chat dan konfigurasi LLM

`POST /api/v1/chat` menerima:

```json
{
  "session_id": "sesi-plp-1",
  "message": "Apa persyaratan mengikuti PLP?",
  "stream": false,
  "retrieval_config": {
    "top_k": 5,
    "score_threshold": 0.65,
    "use_reranker": true,
    "top_k_initial": 15
  }
}
```

Mode JSON mengembalikan `session_id`, `answer`, `rewritten_query`, dan `sources`.
Sumber mencakup teks chunk, metadata, dan skor asli; nomor sitasi `[1]`, `[2]`, dst.
mengikuti urutan sumber. Pertanyaan lanjutan dalam sesi yang sama ditulis ulang
menjadi query mandiri sebelum retrieval. Kegagalan rewriting memakai pesan asli.

`stream` default `true`. Respons `text/event-stream` mengirim event `sources`
(list `point_id`, `score`, `metadata`), event `token` (`{"token": "..."}`), lalu
event `done` (`{"status": "completed"}`). Setiap event diakhiri dua newline.
Header `Cache-Control: no-cache` dan `X-Accel-Buffering: no` membantu pengiriman
langsung melewati proxy. Kegagalan generation sebelum streaming menghasilkan
HTTP 503 pada mode JSON; kegagalan saat SSE menghasilkan event `error` dengan
`status=failed` dan pesan aman, tanpa event `done`. Klien harus memeriksa event ini.

Jika retrieval kosong atau tidak ada skor yang memenuhi threshold, chat menjawab
`Maaf, informasi tersebut tidak ditemukan dalam panduan PLP yang tersedia.`
tanpa memanggil generator LLM. Mode SSE tetap mengirim `sources`, `token`, `done`.
Endpoint retrieve tetap mengembalikan `results: []` dan `total: 0`.

`DELETE /api/v1/chat/{session_id}` menghapus riwayat secara idempotent. Memory
menyimpan enam pesan terakhir per sesi, hingga 1.000 sesi dengan masa kedaluwarsa
idle satu jam dan eviksi LRU. Memory thread-safe dan pasangan user/assistant
disimpan bersama setelah jawaban berhasil selesai. Jawaban parsial yang gagal
atau dibatalkan tidak disimpan. Memory berada di proses aplikasi: restart akan
menghapus riwayat dan beberapa worker tidak berbagi sesi. Gunakan satu worker
untuk MVP; penyimpanan bersama seperti Redis diperlukan untuk banyak worker.

Atur variabel berikut melalui environment atau konfigurasi `.env` Anda:

| Variabel | Perilaku |
| --- | --- |
| `LLM_MODEL` | Model utama, default `gemini-2.5-flash` |
| `GEMINI_API_KEY` / `LLM_API_KEY` | Kunci Gemini; `GEMINI_API_KEY` diprioritaskan |
| `OPENAI_API_KEY` | Mengaktifkan opsi OpenAI |
| `LLM_PROVIDER` | `auto` (default), `gemini`, atau `openai` |
| `OPENAI_MODEL` | Default `gpt-4o-mini`, dipakai OpenAI jika `LLM_MODEL` masih bernama Gemini |
| `LLM_TIMEOUT_SECONDS` | Batas waktu generation, default 30 detik; rewriting maksimal 10 detik |

Pada mode `auto`, model bernama Gemini dengan kunci Gemini memilih Gemini;
selanjutnya kunci OpenAI memilih OpenAI. Gunakan `LLM_PROVIDER` untuk memilih
secara eksplisit jika kedua kunci tersedia, dan sesuaikan nama model dengan
provider tersebut. Client dibuat saat dibutuhkan dan ditutup saat shutdown.
Tanpa kunci LLM, health/retrieve dan chat dengan konteks kosong tetap berjalan;
chat yang memerlukan generation mengembalikan kegagalan layanan yang aman.
Gemini menggunakan API async dari [Google Gen AI SDK](https://googleapis.github.io/python-genai/).

Loguru mencatat `Query Embedding`, `Qdrant Retrieval`, `Reranking` (bila aktif),
dan `LLM Generation` dalam ms, termasuk ketika dependency gagal. Structured
fields `stage` dan `latency_ms` tersedia untuk log sink. Waktu embedding/reranking
mencakup antrean lock dan inisialisasi model; waktu generation streaming mencakup
keseluruhan iterasi stream. Isi chat dan kredensial tidak dicatat oleh service.

## Pengujian

```powershell
uv run pytest
```

Unit test menggunakan mock Qdrant dan model sehingga tidak memerlukan kredensial,
akses jaringan, atau unduhan model. Test mencakup lifecycle, validasi koleksi,
embedding, filter, metadata, reranking, memory concurrent/TTL, rewriting,
adapter Gemini/OpenAI, JSON/SSE, timeout, pembatalan stream, latency, endpoint,
dan kegagalan dependency. Client LLM di-mock agar pytest tidak memanggil API luar.

## Setup untuk kolaborator (pip)

Persyaratan: Python 3.12+, akses ke Qdrant yang **sudah berisi** koleksi capstone,
dan API key Gemini atau OpenAI untuk menghasilkan jawaban. Qdrant lokal yang
kosong belum bisa digunakan: startup memerlukan koleksi yang sudah tersedia.
Minta URL dan akses Qdrant kepada pemilik proyek melalui kanal yang aman.

Dari direktori `ai-service`, jalankan di PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env: atur QDRANT_URL, QDRANT_API_KEY, dan GEMINI_API_KEY.
python -m uvicorn main:app --reload
```

Untuk OpenAI, atur `LLM_PROVIDER=openai` dan isi `OPENAI_API_KEY`. Gunakan URL
Qdrant yang dapat dijangkau dari komputer masing-masing; `localhost` menunjuk
ke komputer yang menjalankan service. Saat pertama dipakai, model embedding dan
reranker opsional diunduh; instalasi dan unduhan model membutuhkan akses internet
serta ruang penyimpanan. Jalankan `python -m pytest` untuk pengujian unit.

Kontrak untuk integrasi tersedia di [API_CONTRACT.md](API_CONTRACT.md); Swagger
interaktif tersedia di `/docs` setelah aplikasi berjalan. Jangan kirim `.env`,
API key, `.venv`, atau data Qdrant lewat Git. Berikan kredensial secara terpisah.
