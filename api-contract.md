# Kontrak API AI Service

Base URL lokal: `http://127.0.0.1:8000`. JSON menggunakan `Content-Type: application/json`. Dokumentasi interaktif dan skema aktual tersedia di `/docs` dan `/openapi.json` saat service berjalan. Endpoint berikut tidak menerapkan autentikasi; atur akses pada lingkungan deployment.

## GET /health

Memeriksa koneksi dan konfigurasi koleksi Qdrant. Berhasil (`200`):

```json
{"status":"ok","qdrant_status":"connected","collection_name":"plp_indoe5_cs96_ov32","points_count":677,"vector_dimension":384}
```

Angka `points_count` adalah contoh dan mengikuti koleksi. Jika pemeriksaan gagal, respons `503` berisi `status: "degraded"`, `qdrant_status: "disconnected"`, serta `points_count` dan `vector_dimension` bernilai `null`. Bila Qdrant atau koleksi tidak tersedia saat startup, aplikasi gagal memulai dan endpoint belum dapat diakses.

## POST /api/v1/retrieve

Mencari chunk tanpa memanggil LLM.

```json
{"query":"Apa syarat mengikuti PLP?","top_k":5,"score_threshold":0.5,"filters":{},"use_rerank":false,"top_k_initial":15}
```

Respons `200` (metadata dan nilai skor merupakan contoh):

```json
{"query":"Apa syarat mengikuti PLP?","results":[{"point_id":"123","score":0.82,"text":"Isi chunk sumber...","metadata":{"page":4,"source_document":"Panduan PLP"},"rerank_score":null}],"total":1}
```

Jika tidak ada hasil: `results: []` dan `total: 0`. `score` adalah skor cosine Qdrant; `rerank_score` adalah skor terpisah ketika reranking digunakan. Metadata mengikuti payload aktual koleksi.

## POST /api/v1/chat

`session_id` mengidentifikasi sesi; gunakan ID yang unik untuk setiap percakapan. Contoh mode JSON:

```json
{"session_id":"sesi-123","message":"Apa syarat mengikuti PLP?","stream":false,"retrieval_config":{"top_k":5,"score_threshold":0.5,"use_reranker":false}}
```

Respons `200`:

```json
{"session_id":"sesi-123","answer":"Jawaban berdasarkan dokumen [1].","rewritten_query":"Apa syarat mengikuti PLP?","sources":[{"point_id":"123","score":0.82,"text":"Isi chunk sumber...","metadata":{"page":4},"rerank_score":null}]}
```

Nomor sitasi merujuk urutan `sources`. `rewritten_query` adalah query yang digunakan untuk retrieval setelah penulisan ulang jika ada riwayat; jika rewriting gagal, pesan asli digunakan. Jawaban dan sumber di atas adalah contoh bentuk respons, bukan isi dokumen sebenarnya. Jika tidak ada konteks, `sources` kosong dan jawaban adalah `Maaf, informasi tersebut tidak ditemukan dalam panduan PLP yang tersedia.`

### Streaming SSE

`stream` default `true`. Dengan `stream: true`, respons bertipe `text/event-stream`. Event dikirim secara berurutan, masing-masing dipisahkan oleh satu baris kosong:

```text
event: sources
data: [{"point_id":"123","score":0.82,"metadata":{"page":4}}]

event: token
data: {"token":"Jawaban "}

event: token
data: {"token":"[1]."}

event: done
data: {"status":"completed"}

```

Token adalah potongan teks dari provider, tidak selalu satu kata. `sources` SSE berisi ID, skor, dan metadata; teks chunk tersedia pada respons JSON atau endpoint retrieve. Jika generation gagal setelah streaming dimulai, server mengirim `event: error` dengan `data: {"status":"failed","message":"Layanan pembuat jawaban sementara tidak tersedia. Silakan coba lagi."}` dan tidak mengirim `done`. Klien perlu membaca event ini. Kegagalan sebelum stream dimulai dapat menghasilkan respons HTTP error biasa.

Contoh PowerShell dengan curl asli:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" -d '{\"session_id\":\"demo\",\"message\":\"Apa itu PLP?\",\"stream\":true}'
```

## DELETE /api/v1/chat/{session_id}

Menghapus riwayat sesi; juga berhasil jika sesi tidak ada.

```json
{"session_id":"sesi-123","status":"cleared"}
```

Memory hanya berada di proses service, kedaluwarsa setelah satu jam idle, dan hilang ketika service restart. Beberapa worker tidak berbagi memory.

## Parameter retrieval

Parameter ini berlaku pada body retrieve atau objek `retrieval_config` pada chat. Objek `retrieval_config` boleh dihilangkan.

| Field | Default | Ketentuan |
| --- | --- | --- |
| `top_k` | 5 | 1–100, batas hasil akhir jika `top_k_final` tidak diisi |
| `score_threshold` | 0.5 | -1 sampai 1; diterapkan sebelum reranking |
| `filters` | null | Filter payload sederhana atau filter Qdrant native |
| `use_rerank` | false | Boolean; `use_reranker` juga diterima sebagai alias |
| `top_k_initial` | 15 | 1–1000; jumlah kandidat jika reranking aktif |
| `top_k_final` | null | 1–100, hanya untuk reranking; `top_k_initial` harus minimal sebesar batas hasil akhir |

Contoh filter sederhana: `{"page":[2,3]}`. Contoh filter native: `{"must":[{"key":"page","range":{"gte":2,"lte":5}}]}`. Dukungan filter bergantung pada payload dan konfigurasi Qdrant.

`query` dan `message` wajib berisi 1–10000 karakter setelah whitespace dihapus; `session_id` wajib berisi 1–200 karakter. Field tambahan ditolak.

## Error HTTP

- `422`: request atau parameter tidak valid; detail validasi mengikuti format FastAPI.
- `503`: dependency retrieval atau LLM tidak tersedia. Response JSON memakai field `detail` dengan pesan aman.
- Streaming dapat mengembalikan HTTP `200` terlebih dahulu lalu event `error`; periksa isi stream, bukan hanya status HTTP.

Service perlu akses ke koleksi Qdrant yang sudah ada. Embedding dan reranker dimuat saat pertama digunakan. Panggilan LLM hanya diperlukan ketika chat memiliki konteks untuk dijawab atau melakukan query rewriting.

---

## Rincian kontrak untuk integrator

**Versi aplikasi:** `0.1.0`. Semua nilai dan isi jawaban pada contoh adalah ilustrasi. Dokumentasi ini menjelaskan perilaku service saat ini; OpenAPI yang dibuat aplikasi tersedia di `/openapi.json` dan Swagger UI di `/docs` saat service berjalan. Body POST harus menggunakan `Content-Type: application/json`. String input di-trim dan field tambahan pada request serta `retrieval_config` ditolak.

### Ringkasan endpoint

| Method | Path | Kegunaan | Respons sukses |
| --- | --- | --- | --- |
| GET | `/health` | Memeriksa koneksi dan konfigurasi koleksi Qdrant | `200`, JSON |
| POST | `/api/v1/retrieve` | Mencari chunk tanpa memanggil LLM | `200`, JSON |
| POST | `/api/v1/chat` | Chat dengan konteks retrieval | `200`, JSON atau SSE |
| DELETE | `/api/v1/chat/{session_id}` | Menghapus riwayat sesi | `200`, JSON |

### Field respons health

| Field | Tipe | Deskripsi |
| --- | --- | --- |
| `status` | string | `ok` jika probe berhasil, `degraded` jika gagal. |
| `qdrant_status` | string | `connected` atau `disconnected`; nilai terakhir juga mencakup kegagalan validasi konfigurasi. |
| `collection_name` | string | Nama koleksi yang dikonfigurasi. |
| `points_count` | integer atau null | Jumlah point yang dilaporkan Qdrant; null jika probe gagal. |
| `vector_dimension` | integer atau null | Dimensi vector koleksi; null jika probe gagal. |

Contoh respons health `503`:

```json
{
  "status": "degraded",
  "qdrant_status": "disconnected",
  "collection_name": "plp_indoe5_cs96_ov32",
  "points_count": null,
  "vector_dimension": null
}
```

### Request dan respons retrieve

| Field request | Tipe | Wajib | Batas |
| --- | --- | --- | --- |
| `query` | string | Ya | 1–10.000 karakter setelah trim. |
| `top_k`, `score_threshold`, `filters`, `use_rerank`, `top_k_initial`, `top_k_final` | Beragam | Tidak | Lihat tabel parameter retrieval di atas. |

| Field respons | Tipe | Deskripsi |
| --- | --- | --- |
| `query` | string | Query setelah trim. |
| `results` | array | Objek chunk yang lolos filter dan threshold. |
| `total` | integer | Jumlah objek dalam `results`. |

Setiap chunk pada `results` maupun `sources` dalam respons chat JSON memiliki bentuk berikut:

| Field | Tipe | Deskripsi |
| --- | --- | --- |
| `point_id` | string | ID point Qdrant sebagai string. |
| `score` | number | Skor cosine Qdrant, sebelum reranking. |
| `text` | string | Teks chunk dari payload yang dikonfigurasi. |
| `metadata` | object | Field payload selain teks; struktur mengikuti koleksi aktual. |
| `rerank_score` | number atau null | Skor reranker apabila digunakan; null jika tidak. |

Teks chunk yang kosong atau tidak valid dilewati. Skor threshold diterapkan sebelum reranking. Jika `use_rerank` aktif, `top_k_initial` harus sekurangnya sama dengan `top_k_final`, atau dengan `top_k` jika `top_k_final` tidak diisi. Gunakan hanya salah satu dari alias `use_rerank` dan `use_reranker`.

Filter sederhana dengan beberapa key menggunakan kondisi AND. Nilai string, integer, atau boolean dicocokkan sebagai nilai tunggal; array tidak kosong dengan elemen string atau integer sejenis mencocokkan salah satu nilai. Float dipetakan ke rentang dengan batas bawah dan atas yang sama. Filter native Qdrant juga didukung; ketersediaannya bergantung pada payload dan konfigurasi indeks Qdrant.

### Request dan respons chat

| Field request | Tipe | Wajib | Default | Deskripsi |
| --- | --- | --- | --- | --- |
| `session_id` | string | Ya | — | ID percakapan, 1–200 karakter setelah trim. |
| `message` | string | Ya | — | Pesan pengguna, 1–10.000 karakter setelah trim. |
| `stream` | boolean | Tidak | `true` | `false` mengembalikan JSON; `true` mengembalikan SSE. |
| `retrieval_config` | object | Tidak | Default retrieval | Parameter retrieval yang dijelaskan di atas. |

| Field respons JSON | Tipe | Deskripsi |
| --- | --- | --- |
| `session_id` | string | ID sesi pada request setelah trim. |
| `answer` | string | Jawaban LLM atau pesan fallback. |
| `rewritten_query` | string | Query yang dipakai untuk retrieval; bila rewriting gagal, pesan asli digunakan. |
| `sources` | array | Chunk yang diberikan sebagai konteks generator, berurutan sesuai nomor sitasi. |

Prompt meminta LLM menyertakan sitasi sesuai urutan `sources`, tetapi service belum memverifikasi kesesuaian tiap klaim atau sitasi. Daftar `sources` adalah konteks yang diberikan kepada LLM dan tidak menjamin setiap sumber dikutip.

Contoh fallback ketika tidak ada chunk:

```json
{
  "session_id": "sesi-123",
  "answer": "Maaf, informasi tersebut tidak ditemukan dalam panduan PLP yang tersedia.",
  "rewritten_query": "Apa syarat mengikuti PLP?",
  "sources": []
}
```

Pesan fallback juga dapat muncul ketika generator mengembalikan teks kosong atau menyatakan konteks tidak cukup; dalam kondisi tersebut `sources` dapat berisi data.

### Rincian event SSE

Respons streaming menggunakan `Content-Type: text/event-stream`, `Cache-Control: no-cache`, dan `X-Accel-Buffering: no`. Setiap event diakhiri baris kosong. Gabungkan nilai `token` secara berurutan.

| Event | Struktur `data` | Arti |
| --- | --- | --- |
| `sources` | Array objek `{point_id, score, metadata}` | Dikirim pertama; tidak memuat `text` atau `rerank_score`. |
| `token` | `{"token":"..."}` | Sepotong teks; tidak harus satu kata. |
| `done` | `{"status":"completed"}` | Jawaban selesai. |
| `error` | `{"status":"failed","message":"..."}` | Generation gagal setelah streaming dimulai; tidak ada event `done`. |

Jika retrieval kosong, `sources` adalah array kosong dan `token` berisi fallback, lalu `done`. Kegagalan retrieval sebelum streaming dimulai menghasilkan `503` JSON biasa. Respons SSE tidak menyediakan `rewritten_query` atau `session_id` sebagai event tersendiri.

### Error dan respons

| Status | Situasi | Bentuk |
| --- | --- | --- |
| `422` | Validasi body, parameter, filter, atau kombinasi reranking gagal. | `detail` berupa array error FastAPI. |
| `503` | Dependency retrieval gagal. | `detail` string. |
| `503` | Generation LLM gagal pada mode JSON. | `detail` string. |
| `503` | Probe health gagal setelah aplikasi dimulai. | Objek health `degraded`. |

Contoh `422`; detail dapat berbeda menurut jenis error dan versi library:

```json
{
  "detail": [
    {"type":"string_too_short","loc":["body","query"],"msg":"String should have at least 1 character","input":""}
  ]
}
```

Contoh `503` retrieval:

```json
{"detail":"Retrieval service is temporarily unavailable"}
```

Contoh `503` generation JSON:

```json
{"detail":"Layanan pembuat jawaban sementara tidak tersedia. Silakan coba lagi."}
```

Jika generation SSE gagal setelah header terkirim, status HTTP dapat tetap `200`. Periksa event `error` dan pastikan `done` diterima sebelum menganggap stream selesai dengan sukses.

### Perilaku sesi dan pengujian

Memory menyimpan sampai enam pesan per sesi dan 1.000 sesi, dengan expiry setelah satu jam idle serta eviksi sesi paling lama tidak dipakai. Memory berada di satu proses: restart menghapusnya dan beberapa worker tidak berbagi riwayat. Pesan dan jawaban disimpan setelah jawaban sukses atau event `done`; stream yang gagal atau dibatalkan tidak disimpan. Request bersamaan untuk satu sesi tidak dijamin berurutan.

Urutan uji yang disarankan: `GET /health`, retrieve, chat JSON, chat lanjutan memakai session ID yang sama, chat SSE, lalu hapus sesi. Periksa isi chunk dan metadata terhadap notebook dengan parameter pencarian yang setara. Koleksi Postman tersedia di [Capstone-PLP.postman_collection.json](ai-service/Capstone-PLP.postman_collection.json).
