import time
import uuid
import numpy as np
from typing import List, Optional
from app.config import get_settings
from app.logger import get_logger
from app.models.schemas import PredictionResult, Label

settings = get_settings()
logger = get_logger(__name__)

_tokenizer = None
_ort_session = None
_pipeline = None
_model_runtime = "not_loaded"


async def load_model():
    """Load model at startup — ONNX preferred, PyTorch pipeline fallback."""
    global _tokenizer, _ort_session, _pipeline, _model_runtime

    logger.info(f"Loading model: {settings.MODEL_NAME}")

    try:
        from transformers import AutoTokenizer
        # Rust-backed HuggingFace fast tokenizer for high-throughput tokenization
        _tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        logger.info("Tokenizer loaded (Rust-backed HuggingFace fast tokenizer)")
    except Exception as e:
        logger.error(f"Tokenizer load failed: {e}")
        raise

    if settings.USE_ONNX:
        await _try_load_onnx()
    else:
        await _load_pytorch_pipeline()

    logger.info(f"Model ready | runtime={_model_runtime}")


async def _try_load_onnx():
    """Attempt ONNX Runtime load with graph optimizations; fall back to PyTorch."""
    global _ort_session, _model_runtime
    import os

    try:
        import onnxruntime as ort

        onnx_path = os.path.join(settings.MODEL_PATH, "model.onnx")
        if not os.path.exists(onnx_path):
            logger.warning(f"ONNX model not found at {onnx_path}, falling back to PyTorch")
            await _load_pytorch_pipeline()
            return

        sess_opts = ort.SessionOptions()
        # Enable all graph-level compiler optimizations (memory scheduling + compute fusion)
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = settings.WORKER_COUNT

        _ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _model_runtime = "onnx"
        logger.info(f"ONNX session loaded | providers={_ort_session.get_providers()}")

    except ImportError:
        logger.warning("onnxruntime not installed, falling back to PyTorch pipeline")
        await _load_pytorch_pipeline()
    except Exception as e:
        logger.error(f"ONNX load error: {e}, falling back to PyTorch pipeline")
        await _load_pytorch_pipeline()


async def _load_pytorch_pipeline():
    """Load HuggingFace PyTorch inference pipeline (CPU)."""
    global _pipeline, _model_runtime
    from transformers import pipeline

    _pipeline = pipeline(
        "text-classification",
        model=settings.MODEL_NAME,
        tokenizer=_tokenizer,
        device=-1,
        return_all_scores=True,
    )
    _model_runtime = "pytorch"
    logger.info("PyTorch pipeline loaded (CPU)")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax — linear algebra: e^x / Σe^x."""
    exp_shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def _run_onnx_inference(texts: List[str]) -> List[dict]:
    """ONNX Runtime batch inference with memory-efficient dynamic padding."""
    inputs = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=settings.MAX_SEQUENCE_LENGTH,
        return_tensors="np",
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

    logits = _ort_session.run(["logits"], ort_inputs)[0]
    probs = _softmax(logits)

    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    results = []
    for prob in probs:
        scores = {label_map[i]: float(p) for i, p in enumerate(prob)}
        best = max(scores, key=scores.get)
        results.append({"label": best, "scores": scores})
    return results


def _run_pytorch_inference(texts: List[str]) -> List[dict]:
    """PyTorch pipeline batch inference."""
    outputs = _pipeline(texts, truncation=True, max_length=settings.MAX_SEQUENCE_LENGTH)
    results = []
    for out in outputs:
        if isinstance(out, list):
            scores = {item["label"]: float(item["score"]) for item in out}
        else:
            scores = {out["label"]: float(out["score"])}
        best = max(scores, key=scores.get)
        results.append({"label": best, "scores": scores})
    return results


async def predict_batch(
    texts: List[str],
    request_id: Optional[str] = None,
) -> tuple[List[PredictionResult], float]:
    """Memory-aware dynamic batching: processes in sub-batches to bound peak memory."""
    if not request_id:
        request_id = str(uuid.uuid4())[:8]

    all_results: List[PredictionResult] = []
    total_start = time.perf_counter()

    for i in range(0, len(texts), settings.MAX_BATCH_SIZE):
        sub_batch = texts[i : i + settings.MAX_BATCH_SIZE]
        batch_start = time.perf_counter()

        if _model_runtime == "onnx" and _ort_session:
            raw = _run_onnx_inference(sub_batch)
        else:
            raw = _run_pytorch_inference(sub_batch)

        batch_ms = (time.perf_counter() - batch_start) * 1000

        for text, result in zip(sub_batch, raw):
            label_str = result["label"].upper()
            scores = {k.upper(): round(v, 4) for k, v in result["scores"].items()}
            confidence = scores.get(label_str, max(scores.values()))
            try:
                label = Label(label_str)
            except ValueError:
                label = Label.POSITIVE

            all_results.append(
                PredictionResult(
                    label=label,
                    confidence=round(confidence, 4),
                    scores=scores,
                    text_length=len(text),
                    inference_ms=round(batch_ms / len(sub_batch), 2),
                )
            )

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)
    logger.info(
        f"Inference | count={len(texts)} | runtime={_model_runtime} | total_ms={total_ms}"
    )
    return all_results, total_ms


def get_model_runtime() -> str:
    return _model_runtime


def get_model_status() -> str:
    if _model_runtime == "onnx" and _ort_session:
        return "ready"
    if _model_runtime == "pytorch" and _pipeline:
        return "ready"
    return "not_loaded"
