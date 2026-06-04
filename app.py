"""
Hindi Voice-to-Voice RAG — Hugging Face Spaces
Pipeline: ASR → FAISS retrieval → (optional) Gemma answer → Piper/gTTS
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from faster_whisper import WhisperModel
from gtts import gTTS
from sentence_transformers import SentenceTransformer

# Hugging Face free CPU tier: keep False. GPU Space: set True or USE_LLM=1 in Space variables.
USE_LLM = False
if os.environ.get("USE_LLM", "").strip().lower() in ("1", "true", "yes"):
    USE_LLM = True
GEMMA_MODEL_ID = "google/gemma-2b-it"
GEMMA_MAX_NEW_TOKENS = 128
GEMMA_GENERATION_TIMEOUT_SEC = 45

WHISPER_MODEL = "medium"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE = "hi"

EMBEDDER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVAL_TOP_K = 2

# Primary ID requested; not in rhasspy/piper-voices catalog (404) — auto-falls back to rohan.
PIPER_VOICE_PRIMARY = "hi_IN-hindi_ldots-medium"
PIPER_VOICE_FALLBACK = "hi_IN-rohan-medium"
PIPER_VOICES_DIR = Path(os.environ.get("PIPER_VOICES_DIR", "piper_voices"))

RESPONSE_AUDIO_PATH = "response.wav"
RESPONSE_MP3_PATH = "response.mp3"

PROMPT_TEMPLATE = (
    "नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर हिंदी में दीजिए।\n"
    "संदर्भ: {context}\n"
    "प्रश्न: {query}\n"
    "उत्तर:"
)

SAMPLE_QUESTIONS = [
    "वाराणसी में सबसे प्रसिद्ध घाट कौन सा है?",
    "गंगा आरती कहाँ होती है?",
    "हनुमान जी ने क्या किया?",
    "बीएचयू कहाँ स्थित है?",
    "सारनाथ क्या है?",
    "प्रयागराज में कौन सी नदियाँ मिलती हैं?",
    "गंगा नदी कहाँ से निकलती है?",
    "अयोध्या किससे जुड़ी है?",
]

# --- Knowledge base (50+ sentences, pure Devanagari) ---
CONTEXTS = [
    # Varanasi ghats
    "वाराणसी का सबसे प्रसिद्ध और मुख्य घाट दशाश्वमेध घाट है।",
    "वाराणसी में सबसे प्रसिद्ध घाट का नाम दशाश्वमेध घाट है।",
    "वाराणसी में सबसे प्रसिद्ध घाट दशाश्वमेध घाट है, जहाँ प्रतिदिन गंगा आरती होती है।",
    "गंगा आरती वाराणसी में हर शाम दशाश्वमेध घाट पर होती है।",
    "मणिकर्णिका घाट वाराणसी का सबसे पवित्र श्मशान घाट है।",
    "अस्सी घाट वाराणसी के दक्षिणी छोर पर स्थित है और यहाँ सूर्योदय दृश्य प्रसिद्ध है।",
    "वाराणसी में चौरासी घाट गंगा नदी के किनारे स्थित हैं।",
    "राज घाट पर महात्मा गांधी का स्मारक स्थित है।",
    "तुलसी घाट पर तुलसीदास जी का मंदिर है।",
    "हरिश्चंद्र घाट वाराणसी का एक प्राचीन और पवित्र घाट है।",
    # Varanasi city / culture
    "वाराणसी को काशी और बनारस के नाम से भी जाना जाता है — यह भारत के सबसे पुराने शहरों में से एक है।",
    "काशी विश्वनाथ मंदिर वाराणसी में स्थित है और भगवान शिव को समर्पित है।",
    "वाराणसी का रेशमी कपड़ा और बनारसी साड़ी विश्व प्रसिद्ध है।",
    "काशी में मरने वाले को मोक्ष मिलता है — यह हिंदू धार्मिक मान्यता है।",
    "वाराणसी में गर्मियों में तापमान पैंतालीस डिग्री सेल्सियस तक पहुँच सकता है।",
    "वाराणसी में मानसून जून से सितंबर तक रहता है।",
    # Ramayana
    "हनुमान जी भगवान राम के परम भक्त और सेवक थे।",
    "हनुमान जी ने लंका में जाकर माता सीता की खोज की और उन्हें राम जी का संदेश दिया।",
    "हनुमान जी ने संजीवनी बूटी लाकर लक्ष्मण जी की जान बचाई थी।",
    "रावण ने माता सीता का अपहरण किया और उन्हें लंका ले गया।",
    "भगवान राम ने वानर सेना की मदद से रावण का वध किया।",
    "अयोध्या भगवान राम की जन्मभूमि है और यह उत्तर प्रदेश में स्थित है।",
    "रामचरितमानस की रचना तुलसीदास ने वाराणसी में की थी।",
    "रामायण में भगवान राम, माता सीता और लक्ष्मण जी की कथा वर्णित है।",
    "भरत ने अयोध्या का राज्य राम जी के नाम पर संभाला था।",
    # Ganga river
    "गंगा नदी हिमालय के गंगोत्री ग्लेशियर से निकलती है।",
    "गंगा नदी को भारत की सबसे पवित्र नदी माना जाता है।",
    "गंगा नदी वाराणसी, प्रयागराज और हरिद्वार जैसे पवित्र शहरों से होकर बहती है।",
    "गंगा का जल भारतीय संस्कृति और धर्म में विशेष महत्व रखता है।",
    "गंगोत्री उत्तराखंड में गंगा की उत्पत्ति स्थल है।",
    # BHU
    "वाराणसी में बनारस हिंदू विश्वविद्यालय यानी बीएचयू स्थित है।",
    "बीएचयू यानी बनारस हिंदू विश्वविद्यालय वाराणसी का सबसे प्रसिद्ध विश्वविद्यालय है।",
    "वाराणसी में स्थित बनारस हिंदू विश्वविद्यालय एशिया के सबसे बड़े विश्वविद्यालयों में से एक है।",
    "बनारस हिंदू विश्वविद्यालय की स्थापना पंडित मदन मोहन मालवीय ने की थी।",
    "बीएचयू का मुख्य परिसर वाराणसी के लंका क्षेत्र में है।",
    # Sarnath
    "सारनाथ वाराणसी के पास स्थित है जहाँ भगवान बुद्ध ने पहला उपदेश दिया था।",
    "सारनाथ में धम्मेक स्तूप और अशोक स्तंभ प्रसिद्ध हैं।",
    "सारनाथ बौद्ध धर्म के प्रमुख तीर्थ स्थलों में से एक है।",
    # Prayagraj
    "प्रयागराज में गंगा, यमुना और सरस्वती नदियों का संगम होता है।",
    "प्रयागराज का पूर्व नाम इलाहाबाद था।",
    "कुंभ मेला प्रयागराज के संगम पर लगभग बारह वर्षों में एक बार आयोजित होता है।",
    "इलाहाबाद विश्वविद्यालय भारत के सबसे पुराने विश्वविद्यालयों में से एक है।",
    # UP geography
    "उत्तर प्रदेश भारत का सबसे अधिक जनसंख्या वाला राज्य है।",
    "उत्तर प्रदेश की राजधानी लखनऊ है।",
    "उत्तर प्रदेश की सीमा नेपाल, उत्तराखंड, बिहार, मध्य प्रदेश और राजस्थान से लगती है।",
    "गंगा-यमुना का मैदान उत्तर प्रदेश का प्रमुख भौगोलिक क्षेत्र है।",
    "वाराणसी, प्रयागराज, लखनऊ और आगरा उत्तर प्रदेश के प्रमुख शहर हैं।",
    # Indian history basics
    "1857 का प्रथम स्वतंत्रता संग्राम मेरठ से शुरू हुआ था।",
    "भारत को १५ अगस्त १९४७ को स्वतंत्रता मिली थी।",
    "महात्मा गांधी ने असहयोग और सविनय अवज्ञा आंदोलन चलाए थे।",
    "अशोक सम्राट ने सारनाथ में धर्मचक्र प्रवर्तन किया था।",
    "मुगल काल में आगरा और फतेहपुर सीकरी महत्वपूर्ण केंद्र थे।",
    "भारतीय संविधान २६ जनवरी १९५० से लागू हुआ था।",
    "डॉ भीमराव अंबेडकर ने संविधान निर्माण में महत्वपूर्ण भूमिका निभाई।",
    "हड़प्पा और मोहनजोदड़ो प्राचीन भारतीय सभ्यता के प्रमुख केंद्र थे।",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("hindi_voice_rag")

# Globals loaded once at startup
whisper_model: WhisperModel | None = None
embedder: SentenceTransformer | None = None
faiss_index: faiss.IndexFlatL2 | None = None
piper_voice = None
gemma_model = None
gemma_tokenizer = None
tts_backend = "none"


def _download_piper_voice(voice_id: str, target_dir: Path) -> Path | None:
    """Download a Piper voice via piper.download_voices; return path to .onnx or None."""
    target_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = target_dir / f"{voice_id}.onnx"
    json_path = target_dir / f"{voice_id}.onnx.json"
    if onnx_path.exists() and json_path.exists():
        return onnx_path
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "piper.download_voices",
                voice_id,
                "--download-dir",
                str(target_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if onnx_path.exists():
            return onnx_path
    except Exception as exc:
        logger.warning("Piper download failed for %s: %s", voice_id, exc)
    return None


def _load_piper_voice():
    """Load Piper TTS; primary voice may be unavailable — fall back to rohan."""
    global piper_voice, tts_backend
    try:
        from piper import PiperVoice
    except ImportError:
        logger.warning("piper-tts not installed; will use gTTS fallback")
        tts_backend = "gtts"
        return

    for voice_id in (PIPER_VOICE_PRIMARY, PIPER_VOICE_FALLBACK):
        onnx_path = _download_piper_voice(voice_id, PIPER_VOICES_DIR)
        if onnx_path is None:
            continue
        try:
            piper_voice = PiperVoice.load(str(onnx_path))
            tts_backend = f"piper ({voice_id})"
            logger.info("Piper TTS loaded: %s", voice_id)
            return
        except Exception as exc:
            logger.warning("Piper load failed for %s: %s", voice_id, exc)

    logger.warning("Piper unavailable; using gTTS fallback")
    tts_backend = "gtts"


def _load_gemma():
    """Load Gemma-2B-IT in 4-bit on GPU; float32 on CPU. Skipped if USE_LLM is False."""
    global gemma_model, gemma_tokenizer
    if not USE_LLM:
        logger.info("USE_LLM=False — skipping Gemma load")
        return

    try:
        import torch
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, token=hf_token)
        if gemma_tokenizer.pad_token is None:
            gemma_tokenizer.pad_token = gemma_tokenizer.eos_token

        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            gemma_model = AutoModelForCausalLM.from_pretrained(
                GEMMA_MODEL_ID,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=hf_token,
            )
            logger.info("Gemma loaded (4-bit, GPU)")
        else:
            gemma_model = AutoModelForCausalLM.from_pretrained(
                GEMMA_MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                token=hf_token,
            )
            logger.info("Gemma loaded (float32, CPU — may be slow on free tier)")
    except Exception as exc:
        logger.error("Gemma load failed: %s", exc)
        gemma_model = None
        gemma_tokenizer = None


def load_models():
    """Load all models once at application startup."""
    global whisper_model, embedder, faiss_index

    logger.info("Loading faster-whisper %s (%s)...", WHISPER_MODEL, WHISPER_COMPUTE_TYPE)
    whisper_model = WhisperModel(WHISPER_MODEL, compute_type=WHISPER_COMPUTE_TYPE)

    logger.info("Loading embedder: %s", EMBEDDER_MODEL)
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    logger.info("Building FAISS index (%d contexts)...", len(CONTEXTS))
    embeddings = np.array(
        embedder.encode(CONTEXTS, normalize_embeddings=True, show_progress_bar=False),
        dtype="float32",
    )
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)

    _load_piper_voice()
    _load_gemma()
    logger.info("Startup complete | TTS: %s | LLM: %s", tts_backend, "on" if gemma_model else "off")


def retrieve(query: str, k: int = RETRIEVAL_TOP_K) -> tuple[list[tuple[str, float]], float]:
    """Return top-k (context, distance) pairs and latency in milliseconds."""
    t0 = time.perf_counter()
    q_emb = np.array(
        embedder.encode([query], normalize_embeddings=True, show_progress_bar=False),
        dtype="float32",
    )
    distances, indices = faiss_index.search(q_emb, k=k)
    latency_ms = (time.perf_counter() - t0) * 1000
    results = [
        (CONTEXTS[int(idx)], float(distances[0][rank]))
        for rank, idx in enumerate(indices[0])
    ]
    return results, latency_ms


def transcribe_audio(audio_path: str) -> tuple[str, float]:
    """Run Whisper ASR with Hindi forced; return text and latency in seconds."""
    t0 = time.perf_counter()
    segments, _info = whisper_model.transcribe(
        audio_path,
        language=WHISPER_LANGUAGE,
        task="transcribe",
    )
    query = " ".join(seg.text.strip() for seg in segments).strip()
    asr_sec = time.perf_counter() - t0
    return query, asr_sec


def generate_answer(query: str, retrieved: list[tuple[str, float]]) -> tuple[str, float, str]:
    """
    Generate Hindi answer with Gemma using retrieved context.
    Returns (answer_text, generation_seconds, source_label).
    """
    if not retrieved:
        return "कोई संदर्भ नहीं मिला।", 0.0, "empty"

    fallback = retrieved[0][0]
    if gemma_model is None or gemma_tokenizer is None:
        return fallback, 0.0, "retrieval_fallback"

    context = "\n".join(ctx for ctx, _ in retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    t0 = time.perf_counter()
    try:
        import torch

        inputs = gemma_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(gemma_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = gemma_model.generate(
                **inputs,
                max_new_tokens=GEMMA_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=gemma_tokenizer.eos_token_id,
            )

        gen_sec = time.perf_counter() - t0
        if gen_sec > GEMMA_GENERATION_TIMEOUT_SEC:
            logger.warning("Gemma exceeded timeout (%.1fs); using retrieval fallback", gen_sec)
            return fallback, gen_sec, "timeout_fallback"

        full_text = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_text.split("उत्तर:")[-1].strip() if "उत्तर:" in full_text else full_text.strip()
        if not answer:
            return fallback, gen_sec, "empty_generation_fallback"
        return answer, gen_sec, "gemma"
    except Exception as exc:
        logger.error("Gemma generation failed: %s", exc)
        return fallback, time.perf_counter() - t0, "error_fallback"


def synthesize_speech(text: str) -> tuple[str | None, str]:
    """Synthesize speech with Piper (offline) or gTTS fallback. Returns (audio_path, backend)."""
    if not text.strip():
        return None, "none"

    if piper_voice is not None:
        try:
            import wave

            with wave.open(RESPONSE_AUDIO_PATH, "wb") as wav_file:
                piper_voice.synthesize_wav(text, wav_file)
            return RESPONSE_AUDIO_PATH, tts_backend
        except Exception as exc:
            logger.warning("Piper synthesis failed: %s — trying gTTS", exc)

    try:
        gTTS(text=text, lang="hi", slow=False).save(RESPONSE_MP3_PATH)
        return RESPONSE_MP3_PATH, "gtts (fallback)"
    except Exception as exc:
        logger.error("gTTS failed: %s", exc)
        return None, "failed"


def _log_pipeline(
    query: str,
    retrieved: list[tuple[str, float]],
    asr_sec: float,
    retrieval_ms: float,
    gen_sec: float,
    total_sec: float,
    answer_source: str,
):
    logger.info("=" * 60)
    logger.info("Query: %s", query)
    logger.info("ASR time: %.3fs | Retrieval: %.1fms | Generation: %.3fs", asr_sec, retrieval_ms, gen_sec)
    logger.info("Total pipeline: %.3fs | Answer source: %s", total_sec, answer_source)
    for rank, (ctx, dist) in enumerate(retrieved, start=1):
        logger.info("  Rank %d | distance=%.4f | %s", rank, dist, ctx)
    logger.info("=" * 60)


def _format_output(
    query: str,
    retrieved: list[tuple[str, float]],
    answer: str,
    asr_sec: float,
    retrieval_ms: float,
    gen_sec: float,
    total_sec: float,
    answer_source: str,
    tts_used: str,
) -> str:
    ctx_lines = "\n".join(
        f"  {i}. (distance: {dist:.3f}) {ctx}"
        for i, (ctx, dist) in enumerate(retrieved, start=1)
    )
    return (
        f"🎙️ Transcribed Question (ASR): {query or '-'}\n\n"
        f"📚 Top-{len(retrieved)} Retrieved Contexts:\n{ctx_lines}\n\n"
        f"💬 Hindi Answer ({answer_source}):\n{answer}\n\n"
        f"⏱️ ASR: {asr_sec:.2f}s | Retrieval: {retrieval_ms:.1f}ms | "
        f"Answer Generation: {gen_sec:.2f}s | Total: {total_sec:.2f}s\n"
        f"🔊 TTS: {tts_used}"
    )


def run_pipeline(query: str, from_audio: bool = False, asr_sec: float = 0.0) -> tuple[str, str | None]:
    """Core RAG pipeline from text query (optionally after ASR)."""
    pipeline_start = time.perf_counter()

    if not query.strip():
        return "⚠️ No question detected. Please speak in Hindi or choose a sample question.", None

    retrieved, retrieval_ms = retrieve(query)
    answer, gen_sec, answer_source = generate_answer(query, retrieved)
    audio_path, tts_used = synthesize_speech(answer)

    total_sec = time.perf_counter() - pipeline_start
    _log_pipeline(query, retrieved, asr_sec, retrieval_ms, gen_sec, total_sec, answer_source)

    text_out = _format_output(
        query, retrieved, answer, asr_sec, retrieval_ms, gen_sec, total_sec, answer_source, tts_used
    )
    return text_out, audio_path


def hindi_pipeline_audio(audio_path: str | None) -> tuple[str, str | None]:
    """Gradio handler: audio in → voice response out."""
    if audio_path is None:
        return "⚠️ No audio received. Please record or upload a Hindi question.", None

    try:
        query, asr_sec = transcribe_audio(audio_path)
        return run_pipeline(query, from_audio=True, asr_sec=asr_sec)
    except Exception as exc:
        logger.error("Pipeline error:\n%s", traceback.format_exc())
        return f"❌ Error: {exc}", None


def hindi_pipeline_text(sample_question: str) -> tuple[str, str | None]:
    """Gradio handler: typed / sample Hindi question (no ASR)."""
    if not sample_question or not sample_question.strip():
        return "⚠️ Please choose a sample Hindi question.", None
    try:
        return run_pipeline(sample_question.strip(), from_audio=False, asr_sec=0.0)
    except Exception as exc:
        logger.error("Text pipeline error:\n%s", traceback.format_exc())
        return f"❌ Error: {exc}", None


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Hindi Voice-to-Voice RAG",
        theme=gr.themes.Soft(primary_hue="orange"),
    ) as demo:
        gr.Markdown(
            """
# 🎙️ Hindi Voice-to-Voice RAG
Record a Hindi question, retrieve grounded context, and hear the answer spoken back in Hindi.

Built by **Vedant Shri Agarwal** · [GitHub Repo](https://github.com/Vedant0527/Hindi-Voice-RAG-demo) · [Hugging Face Space](https://huggingface.co/spaces/Vedant0527/hindi-voice-rag)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎤 Record Hindi Question",
                )
                audio_btn = gr.Button("🚀 Run Audio Query", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 📝 Sample Questions")
                sample_dd = gr.Dropdown(
                    choices=SAMPLE_QUESTIONS,
                    label="Select a Sample Question",
                    value=SAMPLE_QUESTIONS[0],
                )
                text_btn = gr.Button("📖 Run Text Query", variant="secondary")

        with gr.Row():
            text_out = gr.Textbox(label="📋 Pipeline Output", lines=14)
            audio_out = gr.Audio(
                label="🔊 Hindi Answer Audio",
                type="filepath",
                interactive=False,
            )

        gr.Markdown(
            f"""
---
**Stack:** faster-whisper ({WHISPER_MODEL}, {WHISPER_COMPUTE_TYPE}) · MiniLM embeddings · FAISS ·
{'Gemma-2B-IT' if USE_LLM else 'Retrieval-only'} · {tts_backend or 'loading...'}
**Knowledge Base:** {len(CONTEXTS)} Hindi sentences · `USE_LLM={USE_LLM}`
            """
        )

        audio_btn.click(
            fn=hindi_pipeline_audio,
            inputs=audio_in,
            outputs=[text_out, audio_out],
            api_name=False,
            show_api=False,
        )
        text_btn.click(
            fn=hindi_pipeline_text,
            inputs=sample_dd,
            outputs=[text_out, audio_out],
            api_name=False,
            show_api=False,
        )

    return demo


def build_debug_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("Test")
        gr.Textbox()
    return demo


if __name__ == "__main__":
    load_models()
    app = build_debug_ui() if os.environ.get("GRADIO_DEBUG_UI") == "1" else build_ui()
    app.launch(share=True)
