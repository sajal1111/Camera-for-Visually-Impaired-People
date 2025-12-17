# live_internvl.py
import threading
import time
import torch
from PIL import Image
import cv2
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------
# Settings - change these if needed
# -------------------------
MODEL_ID = "OpenGVLab/InternVL2-1B"
SIM_THRESHOLD = 0.6   # similarity threshold for captions
DROIDCAM_URL = "http://100.75.100.107:4747/video"  # change to your stream or use 0 for webcam

# -------------------------
# Device detection
# -------------------------
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

if not use_cuda:
    print("⚠️ CUDA not available: InternVL2-1B requires a CUDA GPU for fp16 inference.")
    print("You can either run a smaller CPU-capable caption model (see instructions) or run this on a machine with an NVIDIA GPU.")
    # We'll still continue, but DO NOT try to load InternVL2 on CPU (it will likely fail / be extremely slow).
else:
    print(f"✅ CUDA is available. Using device: {device}, GPU name:", torch.cuda.get_device_name(0))

# -------------------------
# Shared state
# -------------------------
caption = "Starting..."
lock = threading.Lock()
generating = False

history_text = []
history_embs = []

# -------------------------
# Utility: cosine similarity (normalized)
# -------------------------
def cosine_sim(a, b):
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return torch.dot(a, b).item()

# -------------------------
# Text wrapper for OpenCV overlay
# -------------------------
def wrap_text(text, font, font_scale, thickness, max_width):
    lines = []
    words = text.split()
    current = ""
    for w in words:
        test_line = current + " " + w if current else w
        (width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if width < max_width - 40:
            current = test_line
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines

# -------------------------
# Load models (GPU recommended)
# -------------------------
if use_cuda:
    print("Loading InternVL2 model (this may take a while and needs GPU)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    print("InternVL2 loaded.")
else:
    model = None
    tokenizer = None
    image_processor = None

# Embedding model (sentence-transformers) - runs on CPU if no GPU
embed_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading embedding model (SentenceTransformer) on", embed_device)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=embed_device)

# -------------------------
# Generate caption function
# -------------------------
def generate_caption(frame):
    global caption, generating, history_text, history_embs
    with lock:
        generating = True
    try:
        if model is None:
            # If InternVL2 not available (no GPU), we abort here.
            with lock:
                caption = "InternVL2 model not loaded (GPU required)."
            return

        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess
        pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].half().to("cuda")

        # Ask model to describe
        question = "Describe this image."
        result = model.chat(tokenizer=tokenizer, pixel_values=pixel_values, question=question, generation_config={"max_new_tokens": 50, "do_sample": False})
        new_text = result.strip()

        # Compute embedding (use device of embed_model)
        new_emb = embed_model.encode(new_text, convert_to_tensor=True)
        if new_emb.device.type == 'cpu' and torch.cuda.is_available():
            # Bring to CPU for consistent comparison if history on CPU
            new_emb = new_emb.cpu()

        # Compare with last embeddings
        is_similar = False
        for old_emb in history_embs:
            sim = cosine_sim(new_emb, old_emb)
            if sim >= SIM_THRESHOLD:
                is_similar = True
                break

        if is_similar:
            # Reject and keep current caption unchanged
            return

        # Accept new caption
        with lock:
            caption = new_text

        history_text.append(new_text)
        history_embs.append(new_emb)
        if len(history_text) > 3:
            history_text.pop(0)
            history_embs.pop(0)

    except Exception as e:
        with lock:
            caption = f"Error: {str(e)[:60]}"
    finally:
        with lock:
            generating = False

# -------------------------
# Video stream (DroidCam or webcam)
# -------------------------
def run_loop(url=DROIDCAM_URL):
    cap = cv2.VideoCapture(url)  # use 0 for webcam
    if not cap.isOpened():
        print("❌ Could not open video source:", url)
        return

    print("✅ Connected to video source.")
    frame_count = 0
    global generating

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received, stopping.")
            break

        frame = frame[80:, :]  # crop (same as your original)
        frame_count += 1

        if frame_count % 4 == 0:
            if not generating:
                threading.Thread(target=generate_caption, args=(frame,), daemon=True).start()

        display = frame.copy()
        h, w, _ = display.shape

        with lock:
            current_caption = caption

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        lines = wrap_text(current_caption, font, font_scale, thickness, w)
        line_height = 25
        total_height = line_height * len(lines) + 20

        cv2.rectangle(display, (0, h - total_height), (w, h), (0, 0, 0), -1)

        y = h - total_height + 30
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = int((w - text_size[0]) / 2)
            cv2.putText(display, line, (text_x, y), font, font_scale, (255, 255, 255), thickness)
            y += line_height

        cv2.imshow("InternVL Live Caption", display)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_loop()
