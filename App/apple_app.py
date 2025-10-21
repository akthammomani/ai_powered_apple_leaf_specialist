# app.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch
import torchvision.transforms as T
import base64

# -------------------- Paths --------------------
ART = Path("artifacts")
TS_PATH   = ART / "model.torchscript.pt"
CFG_PATH  = ART / "config_inference.json"
LAB_PATH  = ART / "labels.json"
TEMP_PATH = ART / "temperature.json"  


BANNER   = "header_banner.jpg"
APP_LOGO = "logo 2.jpg"

# -------------------- Page + light styling --------------------
st.set_page_config(
    page_title="AI-Powered Apple Leaf Specialist",
    page_icon=APP_LOGO if Path(APP_LOGO).exists() else "🍎",
    layout="wide",
)

# Card-like look for uploader and camera:
st.markdown("""
<style>
.section { margin-bottom: 1.25rem; }
.section .title { font-size: 1.4rem; font-weight: 700; margin: 0 0 .25rem 0; color: #2c313f; }
.section .sub   { color: #6b7280; margin: 0 0 .75rem 0; }
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]{
  border: 1.5px solid #E6E9EF; background: #F6F8FB; border-radius: 12px; padding: 16px;
}
div[data-testid="stCameraInput"]{
  border: 1.5px solid #E6E9EF; background: #F6F8FB; border-radius: 12px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Sidebar header block */
.sidebar-header {
  display: flex; 
  flex-direction: column; 
  align-items: center; 
  justify-content: center;
  text-align: center;
  gap: .4rem;
  margin-bottom: 1rem;
}
.sidebar-header img {
  width: 96px;              /* tweak size if you like */
  height: auto;
  border-radius: 12px;      /* optional rounded corners */
  box-shadow: 0 2px 6px rgba(0,0,0,.08);
}
.sidebar-title {
  font-weight: 700;
  font-size: 1.0rem;
  line-height: 1.2;
  color: #2c313f;
}
</style>
""", unsafe_allow_html=True)

if Path(BANNER).exists():
    st.image(BANNER, use_column_width=True)
#st.title("AI-Powered Apple Leaf Specialist")
#st.caption("Capture or upload one apple leaf photo. The model predicts healthy · scab · rust · black_rot, or routes to unknown at low confidence.")

# -------------------- Utilities --------------------
def _load_json(p: Path, default):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return default

@st.cache_resource(show_spinner=False)
def load_model_only_ts():
    if not TS_PATH.exists():
        raise FileNotFoundError("artifacts/model.torchscript.pt not found")
    model = torch.jit.load(str(TS_PATH), map_location="cpu").eval()

    cfg  = _load_json(CFG_PATH, {"img_size": 256, "mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]})
    labels = _load_json(LAB_PATH, ["healthy","scab","rust","black_rot"])
    temperature = float(_load_json(TEMP_PATH, {"temperature": 1.0}).get("temperature", 1.0))

    img_size = int(cfg["img_size"])
    mean, std = cfg["mean"], cfg["std"]

    # Match training: resize -> center-crop -> tensor -> normalize
    pad = 32 if img_size >= 224 else int(img_size * 0.125)
    transform = T.Compose([
        T.Resize(img_size + pad),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return model, labels, img_size, temperature, transform

model, labels, IMG_SIZE, TEMPERATURE, transform = load_model_only_ts()

def load_pil(obj) -> Image.Image:
    im = obj if isinstance(obj, Image.Image) else Image.open(obj).convert("RGB")
    return ImageOps.exif_transpose(im)

def predict_probs(pil_img: Image.Image) -> np.ndarray:
    x = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        if TEMPERATURE and TEMPERATURE > 0:
            logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

# -------------------- Quality checks --------------------
def compute_brightness(pil_img: Image.Image) -> float:
    arr = np.asarray(pil_img.resize((256, 256))).astype(np.float32) / 255.0
    y = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
    return float(y.mean())

def green_coverage_soft(pil_img: Image.Image) -> float:
    # HSV band + RGB G-dominance union; tolerant to color casts
    hsv = np.array(pil_img.convert("HSV"))
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    mask_hsv = (H >= 11) & (H <= 85) & (S >= 20) & (V >= 20)
    rgb = np.asarray(pil_img.convert("RGB"))
    R, G, B = rgb[...,0].astype(np.int16), rgb[...,1].astype(np.int16), rgb[...,2].astype(np.int16)
    mask_gdom = (G >= R + 8) & (G >= B + 8)
    return float((mask_hsv | mask_gdom).mean())

def sobel_texture_np(pil_img: Image.Image) -> float:
    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    pad = np.pad(gray, 1, mode="reflect")
    gx = (pad[:-2, :-2]*kx[0,0] + pad[:-2,1:-1]*kx[0,1] + pad[:-2,2:]*kx[0,2] +
          pad[1:-1, :-2]*kx[1,0] + pad[1:-1,1:-1]*kx[1,1] + pad[1:-1,2:]*kx[1,2] +
          pad[2:, :-2]*kx[2,0] + pad[2:,1:-1]*kx[2,1] + pad[2:,2:]*kx[2,2])
    gy = (pad[:-2, :-2]*ky[0,0] + pad[:-2,1:-1]*ky[0,1] + pad[:-2,2:]*ky[0,2] +
          pad[1:-1, :-2]*ky[1,0] + pad[1:-1,1:-1]*ky[1,1] + pad[1:-1,2:]*ky[1,2] +
          pad[2:, :-2]*ky[2,0] + pad[2:,1:-1]*ky[2,1] + pad[2:,2:]*ky[2,2])
    mag = np.hypot(gx, gy)
    return float(mag.var() * 1000.0)

def is_leaf_like(pil_img: Image.Image, cov_min=0.04, cov_max=0.98, tex_min=25.0):
    cov = green_coverage_soft(pil_img)
    tex = sobel_texture_np(pil_img)
    ok = (cov_min <= cov <= cov_max) and (tex >= tex_min)
    return ok, cov, tex

def decide(probs: np.ndarray, labels, threshold: float):
    k = int(np.argmax(probs)); p = float(probs[k])
    return (labels[k], p, k) if p >= threshold else ("unknown", p, k)

# -------------------- Care tips --------------------
CARE = {
    "healthy":[
        "No action required; keep routine scouting weekly.",
        "Light pruning to maintain airflow; remove dense water sprouts.",
        "Irrigate at soil level; avoid wetting foliage late in the day.",
        "Sanitation: rake and remove healthy leaf drop away from trunks."
    ],
    "scab":[
        "Cull infected leaves/fruit; bag and trash (do not compost).",
        "Sanitize pruners between cuts (70% alcohol or 10% bleach).",
        "Apply a registered fungicide early-season or at first signs per local guidance.",
        "Prevention: prune for airflow; remove leaf litter over winter."
    ],
    "rust":[
        "Check nearby juniper/cedar (alternate host); prune galls if feasible.",
        "Use protectant fungicide at tight cluster/pink where rust pressure is high.",
        "Improve airflow (thin dense branches); remove heavily infected leaves.",
        "Avoid overhead irrigation; keep mulch off the trunk flare."
    ],
    "black_rot":[
        "Prune cankers 4–6 inches below visible margins; dispose of prunings.",
        "Remove mummified fruit and infected spurs; sanitize tools.",
        "If orchard had prior black rot pressure, run an early-season fungicide program.",
        "Keep orchard floor clean; remove dead wood where fungus overwinters."
    ],
    "unknown":[
        "Low confidence: retake in bright, even light; fill the frame with one leaf.",
        "Wipe lens; hold phone steady; avoid backlight and deep shadows.",
        "Capture both sides and the most symptomatic area.",
        "If symptoms persist, contact your local extension agent with multiple photos."
    ],
}
def render_care(label):
    st.subheader("Care & Prevention")
    for t in CARE.get(label, CARE["unknown"]):
        st.write("• " + t)
    st.caption("General guidance only. Follow local regulations and product labels for any chemical applications.")

# -------------------- Session state (single-source input) --------------------
if "show_camera" not in st.session_state:   st.session_state.show_camera = False
if "source" not in st.session_state:        st.session_state.source = None  # 'camera'|'upload'
if "captured" not in st.session_state:      st.session_state.captured = None
if "upload" not in st.session_state:        st.session_state.upload = None
if "keep_camera_on" not in st.session_state:st.session_state.keep_camera_on = False

def open_camera():
    st.session_state.show_camera = True
    st.session_state.source = "camera"
    st.session_state.upload = None

def close_camera():
    st.session_state.show_camera = False

def on_upload_change():
    st.session_state.upload = st.session_state.get("uploader")
    st.session_state.source = "upload"
    st.session_state.show_camera = False

# -------------------- Sidebar controls --------------------
def sidebar_logo(title:str, path:str):
    if Path(path).exists():
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        ext = Path(path).suffix.lstrip(".").lower() or "png"
        img_html = f'<img src="data:image/{ext};base64,{b64}" alt="logo" />'
    else:
        img_html = '<div style="font-size:48px">🍎</div>'
    st.markdown(f'''
        <div class="sidebar-header">
            {img_html}
            <div class="sidebar-title">{title}</div>
        </div>
    ''', unsafe_allow_html=True)

with st.sidebar:
    sidebar_logo("AI-Powered Apple Leaf Specialist", APP_LOGO)
    st.subheader("Settings")
    THRESHOLD = st.slider("Decision threshold (τ)", 0.0, 0.99, 0.85, 0.01)
    dark_thr   = st.slider("Too dark threshold", 0.05, 0.50, 0.25, 0.01)
    bright_thr = st.slider("Too bright threshold", 0.50, 0.99, 0.90, 0.01)
    cov_min = st.slider("Min green coverage (camera gate)", 0.00, 0.50, 0.04, 0.01)
    tex_min = st.slider("Min texture score (camera gate)", 0.0, 300.0, 25.0, 1.0)
    st.checkbox("Keep camera open after capture", value=st.session_state.keep_camera_on, key="keep_camera_on")
    st.caption(f"Engine: TorchScript · Temperature: {TEMPERATURE:.2f} · Image size: {IMG_SIZE}")
    st.markdown("**Classes**: " + " · ".join(labels))

# -------------------- Inputs (cards, side-by-side) --------------------
st.subheader("Add a leaf photo")
left, right = st.columns([1,1], gap="large")

with left:
    st.markdown('<div class="section"><div class="title">Upload Photo</div>'
                '<div class="sub">Drop a JPG/PNG here, or browse</div>', unsafe_allow_html=True)
    st.file_uploader(label="", type=["jpg","jpeg","png"], key="uploader", on_change=on_upload_change)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section"><div class="title">Record Photo</div>'
                '<div class="sub">Use your device camera</div>', unsafe_allow_html=True)
    if not st.session_state.show_camera:
        st.button("Open camera", on_click=open_camera)
        cap = None
    else:
        cap = st.camera_input("", key="camera_input")
        if st.button("Close camera"): close_camera()
        if cap is not None:
            st.session_state.captured = cap
            st.session_state.source = "camera"
            if not st.session_state.keep_camera_on:
                close_camera()
    st.markdown("</div>", unsafe_allow_html=True)

# Select active source strictly
file = st.session_state.captured if st.session_state.source == "camera" else (
       st.session_state.upload if st.session_state.source == "upload" else None)

# -------------------- Main inference path --------------------
if file:
    pil = load_pil(file)

    # Brightness check for both sources
    b = compute_brightness(pil)
    if b < dark_thr:
        st.warning(f"Image appears too dark (brightness {b:.2f}). Retake under brighter, even lighting.")
        st.stop()
    if b > bright_thr:
        st.warning(f"Image appears too bright/washed-out (brightness {b:.2f}). Retake avoiding direct glare.")
        st.stop()

    # Leaf-likeness gate ONLY for camera
    if st.session_state.source == "camera":
        bypass_gate = st.checkbox("Bypass leaf check for this camera image", value=False)
        ok_leaf, cov, tex = is_leaf_like(pil, cov_min=cov_min, cov_max=0.98, tex_min=tex_min)
        if not (ok_leaf or bypass_gate):
            st.warning(
                f"This photo might not be a single leaf (green_coverage≈{cov:.2f}, texture≈{tex:.0f}). "
                "Retake: fill the frame with one leaf in even lighting, sharp focus — or tick the bypass to proceed."
            )
            st.stop()

    c1, c2 = st.columns([1,1])
    with c1:
        st.image(pil, caption="Input", use_column_width=True)

    probs = predict_probs(pil)
    pred_label, pred_conf, _ = decide(probs, labels, THRESHOLD)

    order = np.argsort(probs)[::-1]
    df = pd.DataFrame([(labels[i], float(probs[i])) for i in order[:4]],
                      columns=["label","probability"])

    with c2:
        st.metric("Decision", pred_label, delta=f"{pred_conf:.3f}")
        st.dataframe(df.style.format({"probability":"{:.3f}"}), use_container_width=True)

    render_care(pred_label)
    st.caption("Calibrated ResNet-18 (TorchScript). Low-confidence predictions route to ‘unknown’.")
else:
    st.info("Upload a photo or open the camera to begin.")
