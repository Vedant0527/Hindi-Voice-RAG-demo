# 🎙️ Hindi Voice-to-Voice RAG

A multilingual Voice AI system that accepts Hindi speech, retrieves relevant knowledge using semantic search, and responds back in Hindi text and speech.

## 🚀 Overview

Hindi Voice-to-Voice RAG is a low-latency Retrieval-Augmented Generation (RAG) system designed for Hindi-speaking users.

The pipeline converts spoken Hindi into text, retrieves the most relevant knowledge using multilingual embeddings and FAISS vector search, and generates a Hindi response that is spoken back to the user using Piper TTS.

### Pipeline

```text
Hindi Audio
    ↓
Whisper ASR
    ↓
Hindi Text Query
    ↓
Multilingual Embeddings
    ↓
FAISS Retrieval
    ↓
Knowledge Response
    ↓
Hindi Text Output
    ↓
Piper / gTTS
    ↓
Hindi Audio Output
```

---

## ✨ Features

* 🎤 Hindi Speech Recognition using Faster-Whisper
* 🌍 Multilingual Semantic Retrieval
* ⚡ FAISS Vector Search
* 🔊 Hindi Voice Output using Piper TTS
* ☁️ Automatic gTTS Fallback
* 🧠 Optional Gemma-2B Integration
* 🚀 Hugging Face Spaces Ready

---

## 📝 Sample Hindi Questions

Try asking:

* वाराणसी में सबसे प्रसिद्ध घाट कौन सा है?
* गंगा नदी कहाँ से निकलती है?
* सारनाथ का ऐतिहासिक महत्व क्या है?
* बीएचयू की स्थापना किसने की थी?
* रामायण में अयोध्या का क्या महत्व है?
* वाराणसी को काशी क्यों कहा जाता है?
* दशाश्वमेध घाट क्यों प्रसिद्ध है?

---

## 🛠️ Tech Stack

| Component          | Technology            |
| ------------------ | --------------------- |
| Speech Recognition | Faster-Whisper        |
| Embeddings         | Sentence Transformers |
| Vector Database    | FAISS                 |
| Text-to-Speech     | Piper TTS             |
| Fallback TTS       | gTTS                  |
| Optional LLM       | Gemma-2B-IT           |
| Interface          | Gradio                |
| Deployment         | Hugging Face Spaces   |

---

## 🧠 Knowledge Base

The current knowledge base contains curated Hindi content related to:

* Varanasi
* Ganga River
* Sarnath
* Banaras Hindu University (BHU)
* Ramayana Context
* Regional History

---

## ⚙️ Gemma-2B Configuration

By default, the application runs in retrieval-only mode:

```python
USE_LLM = False
```

This setting is recommended for CPU-based deployments and Hugging Face Free Spaces.

To enable full Retrieval-Augmented Generation using Gemma-2B:

```python
USE_LLM = True
```

Recommended deployment:

* NVIDIA T4
* L4
* Any CUDA-enabled GPU

---

## 📂 GitHub Repository

GitHub:

https://github.com/Vedant0527/Hindi-Voice-RAG-demo

---

## 👨‍💻 Author

Built by **Vedant Shri Agarwal**

---

## 📜 License

MIT License
