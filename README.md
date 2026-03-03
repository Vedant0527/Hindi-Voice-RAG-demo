# Hindi-Voice-RAG-demo
Hindi Voice ASR + RAG + TTS
# 🎙️ Hindi Voice-to-Voice RAG — 48-hour Prototype

**Pipeline:** Hindi speech → Whisper ASR → FAISS retrieval → gTTS spoken response

📹 Demo video: (https://www.loom.com/share/9a50eb9c09cf47d8947bfb2cc6910e82)

---

## Results

| Stage | Metric | Value |
|---|---|---|
| ASR | Model | whisper-medium |
| ASR | Latency | 2.2s (Colab T4 GPU) |
| Retrieval | Embedder | paraphrase-multilingual-MiniLM-L12-v2 |
| Retrieval | FAISS latency | 19ms |
| Retrieval | Correct answer returned | ✅ दशाश्वमेध घाट |
| TTS | Hindi audio playback | ✅ gTTS working |

---

## Pipeline
```
Hindi audio → Whisper medium (language=hi) → FAISS semantic search
→ top-1 context retrieved → gTTS Hindi TTS → spoken response
```

## Key Finding

Switching from `all-MiniLM-L6-v2` (English-only) to
`paraphrase-multilingual-MiniLM-L12-v2` was critical —
the English embedder failed to match Whisper's imperfect Hindi
output (वारनसी, प्रसिड) to the correct knowledge base entries
(वाराणसी, प्रसिद्ध). The multilingual model resolved this
immediately, confirming that Hindi RAG requires multilingual
embedders at every stage.

---

## How to Run

1. Open `Hindi_voice_RAG.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Upload any Hindi `.wav` or `.mp3` when prompted
4. Run all cells top to bottom

---

## Screenshots

**Full pipeline output:**
<img width="721" height="458" alt="screenshots:full_pipeline" src="https://github.com/user-attachments/assets/332a21c9-1b6e-4983-8c0d-2ff90bd234b0" />

**WER comparison:**
<img width="623" height="173" alt="screenshots:wer_comparison" src="https://github.com/user-attachments/assets/39073c80-2081-47e4-b093-7f1505c722f8" />

---

## Next Steps

- faster-whisper (INT8) for Raspberry Pi deployment
- Replace gTTS with offline Coqui TTS
- Upgrade to LaBSE embeddings for better Hindi semantics
- Add real LLM generation step (true RAG)
- Stream ASR in real-time chunks

---

*Built by Vedant · March 2026*
