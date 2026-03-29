# 🎙️ Hindi Voice-to-Voice RAG

**Pipeline:** Hindi speech → faster-whisper ASR → FAISS retrieval → gTTS spoken response

🌐 **Live Demo:** https://huggingface.co/spaces/Vedant0527/hindi-voice-rag
💻 **Notebook:** Hindi_voice_RAG.ipynb

---

## Results

| Stage | Metric | Value |
|---|---|---|
| ASR | Model | faster-whisper medium INT8 |
| ASR | Latency | **1.20s** (45% faster than openai-whisper) |
| Retrieval | Embedder | paraphrase-multilingual-MiniLM-L12-v2 |
| Retrieval | FAISS latency | **15.7ms** |
| Retrieval | Correct answer | ✅ दशाश्वमेध घाट |
| TTS | Hindi audio playback | ✅ gTTS working |

---

## Pipeline
```
Hindi audio → faster-whisper medium INT8 (language=hi)
→ FAISS semantic search (multilingual embedder)
→ top-1 context retrieved
→ gTTS Hindi TTS → spoken response
```

---

## Key Finding

Switching from `all-MiniLM-L6-v2` (English-only) to
`paraphrase-multilingual-MiniLM-L12-v2` was critical —
the English embedder failed to match Whisper's imperfect Hindi
output (वारनसी, प्रसिड) to correct knowledge base entries
(वाराणसी, प्रसिद्ध). The multilingual model resolved this
immediately, confirming that **Hindi RAG requires multilingual
embedders at every stage**, not just multilingual ASR.

---

## Tech Stack

| Component | Tool |
|---|---|
| ASR | faster-whisper medium (INT8 quantized) |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector search | FAISS IndexFlatL2 |
| TTS | gTTS (Hindi) |
| Interface | Gradio |
| Runtime | Google Colab T4 GPU |

---

## How to Run

1. Open `Hindi_voice_RAG.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Upload any Hindi `.wav` or `.mp3` when prompted
4. Run all cells top to bottom

Or try the live demo directly:
👉 https://huggingface.co/spaces/Vedant0527/hindi-voice-rag

---

## Screenshots

**Full pipeline output:**
<img width="721" height="458" alt="full_pipeline" src="https://github.com/user-attachments/assets/332a21c9-1b6e-4983-8c0d-2ff90bd234b0" />

**WER comparison:**
<img width="623" height="173" alt="wer_comparison" src="https://github.com/user-attachments/assets/39073c80-2081-47e4-b093-7f1505c722f8" />

---

## Next Steps

- Offline Coqui/Piper TTS (remove internet dependency)
- LaBSE embeddings for better Hindi semantic matching
- Real LLM generation step (Gemma-2B-IT)
- whisper.cpp for Raspberry Pi ARM deployment
- Streaming ASR in real-time chunks

---

*Built by Vedant Shri Agarwal*
*GitHub: github.com/Vedant0527*
