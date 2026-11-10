# 🧪 Panduan Lengkap Pengujian API — Qdrant RAG Chatbot Service

Dokumen ini berisi panduan komprehensif untuk menguji seluruh endpoint pada layanan **Qdrant RAG Chatbot API Service**, lengkap dengan penjelasan parameter, tipe data, skenario uji, serta perintah cURL dan PowerShell yang siap disalin.

---

## 📌 Informasi Dasar API

* **Base URL:** `http://127.0.0.1:8000`
* **Interactive Docs (Swagger UI):** [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)
* **ReDoc UI:** [`http://127.0.0.1:8000/redoc`](http://127.0.0.1:8000/redoc)
* **Koleksi Qdrant Aktif:** `plp_indoe5_cs96_ov32` (677 points, 384 dimensi, Cosine)

### Cara Menjalankan Server API:
Pastikan berada di folder `ai-service`:
```powershell
cd e:\VSCODE\capstone-api\ai-service
python -m uv run uvicorn main:app --reload --port 8000
```

---

## 1. `GET /health` (Cek Status & Koneksi Database)

Memeriksa kesehatan aplikasi dan memvalidasi koneksi langsung ke cluster Qdrant Cloud.

### Parameter:
* *Tidak membutuhkan parameter (No body / query params).*

### Contoh Request (cURL):
```bash
curl -X GET http://127.0.0.1:8000/health
```

### Contoh Request (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get | ConvertTo-Json
```

### Contoh Response (200 OK):
```json
{
  "status": "ok",
  "qdrant_status": "connected",
  "collection_name": "plp_indoe5_cs96_ov32",
  "points_count": 677,
  "vector_dimension": 384
}
```

### Penjelasan Field Response:
| Field | Tipe | Keterangan |
|---|---|---|
| `status` | string | Status service FastAPI (`"ok"`). |
| `qdrant_status` | string | Status koneksi ke Qdrant Cloud (`"connected"` atau `"unreachable"`). |
| `collection_name`| string | Nama koleksi yang aktif (`"plp_indoe5_cs96_ov32"`). |
| `points_count` | integer | Jumlah chunk/vektor yang tersedia di dalam database (677). |
| `vector_dimension` | integer | Dimensi vektor embedding (384). |

---

## 2. `POST /api/v1/retrieve` (Pure Semantic Search & Reranking)

Endpoint untuk menguji pencarian potongan dokumen (*chunks*) di Qdrant tanpa memanggil model LLM. Berguna untuk menguji relevansi data dan kualitas skor kemiripan.

### Parameter Request (JSON Body):

| Parameter | Tipe | Wajib? | Default | Keterangan |
|---|---|:---:|:---:|---|
| `query` | string | **Ya** | - | Kalimat pertanyaan atau kata kunci pencarian (1 - 10.000 karakter). |
| `top_k` | integer | Tidak | `5` | Jumlah potongan dokumen teratas yang ingin diambil (1 - 100). |
| `score_threshold` | float | Tidak | `0.5` | Batas minimal cosine similarity score (-1.0 s/d 1.0). Chunk dengan skor di bawah nilai ini akan diabaikan. |
| `use_rerank` | boolean | Tidak | `false` | Jika `true`, mengaktifkan Two-Stage Retrieval menggunakan **FlashRank Cross-Encoder**. |
| `top_k_initial` | integer | Tidak | `15` | Jumlah kandidat awal yang ditarik dari Qdrant sebelum di-rerank (hanya berlaku jika `use_rerank: true`). |
| `top_k_final` | integer | Tidak | `null` | Jumlah hasil akhir setelah di-rerank. Jika kosong, akan memakai nilai `top_k`. |
| `filters` | object | Tidak | `null` | Filter metadata Qdrant (misal: filter bab, nomor dokumen, atau halaman). |

---

### Contoh 2.1: Pencarian Standar (Baseline Bi-Encoder)

#### cURL:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa saja bentuk rekognisi kegiatan di luar kampus?",
    "top_k": 3,
    "score_threshold": 0.5
  }'
```

#### PowerShell:
```powershell
$body = @{
    query = "Apa saja bentuk rekognisi kegiatan di luar kampus?"
    top_k = 3
    score_threshold = 0.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/retrieve" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 5
```

---

### Contoh 2.2: Pencarian Akurasi Tinggi (Two-Stage dengan Reranker)

Mengambil 15 kandidat awal lalu diurutkan ulang dengan FlashRank untuk memilih 3 terbaik:

#### cURL:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Berapa batas waktu pelaporan magang keprofesian?",
    "use_rerank": true,
    "top_k_initial": 15,
    "top_k_final": 3,
    "score_threshold": 0.3
  }'
```

---

### Contoh 2.3: Pencarian dengan Metadata Filter

Membatasi pencarian hanya pada dokumen tertentu:
```json
{
  "query": "Kewajiban peserta magang",
  "top_k": 3,
  "filters": {
    "doc_index": 2
  }
}
```

---

### Contoh Response `POST /api/v1/retrieve` (200 OK):
```json
{
  "query": "Apa saja bentuk rekognisi kegiatan di luar kampus?",
  "total": 3,
  "results": [
    {
      "point_id": "f1fc1929-3e96-5f19-a98c-82059fb642fb",
      "score": 0.6512,
      "rerank_score": 0.9991,
      "text": "3.4 Bentuk Rekognisi SK2PM\nSelain ke mata kuliah, rekognisi kegiatan pembelajaran di luar kampus dapat dikonversi menjadi poin SK2PM...",
      "metadata": {
        "chapter": "BAB 3 REKOGNISI",
        "chunk_id": "cs96_ov32_000076",
        "doc_index": 1,
        "page": 26,
        "section": "",
        "source_document": "[2025] 1 BUKU PANDUAN PEMBELAJARAN DI LUAR PROGRAM STUDI INSTITUT TEKNOLOGI KALIMANTAN.md"
      }
    }
  ]
}
```

---

## 3. `POST /api/v1/chat` (Endpoint Utama Chatbot RAG)

Endpoint untuk berinteraksi dengan AI Assistant. Mendukung mode respons **JSON Biasa** dan **Server-Sent Events (SSE Streaming)**.

### Parameter Request (JSON Body):

| Parameter | Tipe | Wajib? | Default | Keterangan |
|---|---|:---:|:---:|---|
| `session_id` | string | **Ya** | - | ID unik percakapan user (misal: `"user_123"`, `"sess_abc"`). Digunakan untuk mengingat konteks percakapan (*multi-turn*). |
| `message` | string | **Ya** | - | Pesan atau pertanyaan dari user. |
| `stream` | boolean | Tidak | `true` | **`false`**: Mengembalikan JSON utuh setelah selesai digenerate.<br>**`true`**: Mengirimkan jawaban per kata/token via SSE streaming. |
| `retrieval_config` | object | Tidak | `{}` | Konfigurasi pencarian Qdrant (sama seperti parameter pada `/retrieve`: `top_k`, `score_threshold`, `use_rerank`, `filters`). |

---

### 3.1. Pengujian Mode JSON Biasa (`stream: false`)

Cocok untuk integrasi API backend-to-backend atau pengujian via Postman.

#### cURL:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_demo_01",
    "message": "Apa saja bentuk rekognisi dalam program PLP ITK?",
    "stream": false,
    "retrieval_config": {
      "top_k": 3,
      "use_rerank": true,
      "score_threshold": 0.4
    }
  }'
```

#### PowerShell:
```powershell
$body = @{
    session_id = "user_demo_01"
    message = "Apa saja bentuk rekognisi dalam program PLP ITK?"
    stream = $false
    retrieval_config = @{
        top_k = 3
        use_rerank = $true
        score_threshold = 0.4
    }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 6
```

#### Contoh Response (200 OK):
```json
{
  "session_id": "user_demo_01",
  "answer": "Berdasarkan Buku Panduan PLP ITK, bentuk rekognisi pembelajaran di luar kampus terdiri dari konversi ke mata kuliah dan konversi menjadi poin SK2PM [1]. Untuk rekognisi mata kuliah, bobot maksimal yang dapat diakui adalah 20 SKS per semester [2].",
  "rewritten_query": "Apa saja bentuk rekognisi dalam program PLP ITK?",
  "sources": [
    {
      "point_id": "f1fc1929-3e96-5f19-a98c-82059fb642fb",
      "score": 0.6512,
      "rerank_score": 0.9991,
      "text": "3.4 Bentuk Rekognisi SK2PM...",
      "metadata": {
        "chapter": "BAB 3 REKOGNISI",
        "doc_index": 1,
        "page": 26,
        "source_document": "[2025] 1 BUKU PANDUAN PEMBELAJARAN DI LUAR PROGRAM STUDI INSTITUT TEKNOLOGI KALIMANTAN.md"
      }
    }
  ]
}
```

---

### 3.2. Pengujian Mode Streaming SSE (`stream: true`)

Menghasilkan efek mengetik kata-per-kata secara real-time seperti ChatGPT.

#### cURL (Gunakan flag `-N` untuk mematikan buffer terminal):
```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_demo_02",
    "message": "Bagaimana prosedur pendaftaran magang riset?",
    "stream": true
  }'
```

#### Alur Event SSE yang Diterima Klien:
```text
event: sources
data: [{"point_id": "...", "score": 0.72, "metadata": {"source_document": "...", "page": 12}}]

event: token
data: {"token": "Prosedur "}

event: token
data: {"token": "pendaftaran "}

event: token
data: {"token": "magang "}

event: token
data: {"token": "riset "}

event: done
data: {"status": "completed"}
```

---

### 3.3. Pengujian Multi-Turn Conversation (Riwayat Chat & Rewriting)

Uji apakah sistem mengingat percakapan sebelumnya menggunakan `session_id` yang sama:

#### Pertanyaan 1:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sesi_multiturn",
    "message": "Jelaskan tentang magang keprofesian di ITK",
    "stream": false
  }'
```

#### Pertanyaan 2 (Pertanyaan Lanjutan / Follow-up):
*Perhatikan: user hanya bertanya "Berapa SKS maksimalnya?", sistem akan otomatis menulis ulang menjadi "Berapa SKS maksimal untuk magang keprofesian di ITK?" sebelum mencari ke Qdrant.*
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sesi_multiturn",
    "message": "Berapa SKS maksimalnya?",
    "stream": false
  }'
```

---

### 3.4. Pengujian Guardrail Anti-Halusinasi (Fallback)

Uji dengan pertanyaan di luar domain dokumen panduan PLP ITK:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sesi_guardrail",
    "message": "Siapakah presiden pertama Amerika Serikat?",
    "stream": false
  }'
```

#### Hasil yang Diharapkan:
Sistem mendeteksi bahwa tidak ada dokumen Qdrant yang relevan, sehingga langsung menolak secara sopan **tanpa berhalusinasi**:
```json
{
  "session_id": "sesi_guardrail",
  "answer": "Maaf, informasi tersebut tidak ditemukan dalam panduan PLP yang tersedia.",
  "rewritten_query": "Siapakah presiden pertama Amerika Serikat?",
  "sources": []
}
```

---

## 4. `DELETE /api/v1/chat/{session_id}` (Reset Riwayat Chat)

Menghapus riwayat percakapan sesi tertentu dari memori server.

### Parameter:
* `session_id` (Path Parameter, Wajib): ID sesi yang ingin dihapus.

### Contoh Request (cURL):
```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/chat/sesi_multiturn
```

### Contoh Request (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat/sesi_multiturn" -Method Delete | ConvertTo-Json
```

### Contoh Response (200 OK):
```json
{
  "session_id": "sesi_multiturn",
  "status": "cleared"
}
```

---

## 5. Ringkasan Skenario Uji (Test Checklist)

| No | Skenario | Endpoint | Parameter Kunci | Ekspektasi |
|:---:|---|---|---|---|
| 1 | Cek Konektivitas Database | `GET /health` | - | Status 200, points_count = 677. |
| 2 | Pencarian Chunks Murni | `POST /api/v1/retrieve` | `query: "..."`, `top_k: 3` | Mengembalikan 3 potongan dokumen + metadata asli. |
| 3 | Uji Reranker FlashRank | `POST /api/v1/retrieve` | `use_rerank: true`, `top_k_initial: 15` | Mengembalikan hasil dengan field `rerank_score`. |
| 4 | Tanya-Jawab RAG (JSON) | `POST /api/v1/chat` | `stream: false` | Jawaban jelas + sitasi `[1]` + list `sources`. |
| 5 | Tanya-Jawab RAG (Streaming) | `POST /api/v1/chat` | `stream: true` | Token dikirim per kata melalui SSE stream. |
| 6 | Konteks Multi-turn | `POST /api/v1/chat` | `session_id` sama, tanya follow-up | `rewritten_query` berubah menjadi kalimat utuh. |
| 7 | Guardrail Anti-Halusinasi | `POST /api/v1/chat` | Pertanyaan di luar topik dokumen | Menjawab kalimat penolakan standar secara sopan. |
| 8 | Hapus Sesi Chat | `DELETE /api/v1/chat/{session_id}` | `session_id: "..."` | Status 200, status = "cleared". |

