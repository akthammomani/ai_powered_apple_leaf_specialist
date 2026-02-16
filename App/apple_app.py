# app.py
import json
from pathlib import Path
import base64

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import streamlit.components.v1 as components
import torch
import torchvision.transforms as T

# -------------------- Paths --------------------
ART = Path("Data_Directory/artifacts")
TS_PATH   = ART / "model.torchscript.pt"
CFG_PATH  = ART / "config_inference.json"
LAB_PATH  = ART / "labels.json"
TEMP_PATH = ART / "temperature.json"

BANNER   = "header_banner.jpg"
APP_LOGO = "logo 3.png"

# -------------------- Defaults --------------------
THRESHOLD     = 0.85
dark_thr      = 0.25
bright_thr    = 0.90
PREVIEW_MAX_W = 420
PREVIEW_MAX_H = 420

# Camera leaf-gate defaults (used later)
cov_min = 0.04
tex_min = 25.0

# -------------------- Page config (force sidebar open) --------------------
st.markdown("""
<style>
div.block-container {
  padding-top: 0.5rem !important;   
}

@media (min-width: 992px) {
  div.block-container {
    padding-top: 0.25rem !important;
  }
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="AI-Powered Apple Leaf Specialist",
    page_icon=APP_LOGO if Path(APP_LOGO).exists() else "🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- CSS --------------------
st.markdown(
    """
<style>
:root{
  --sb-w: 420px;
  --sb-bg: #F3F4F6;
  --card-bg: #F3F4F6;
  --border: #E5E7EB;
  --text: #111827;
  --muted: #6B7280;
}

section[data-testid="stSidebar"]{
  width: var(--sb-w) !important;
  min-width: var(--sb-w) !important;
  background: var(--sb-bg) !important;
}
section[data-testid="stSidebar"] > div{
  width: var(--sb-w) !important;
  background: var(--sb-bg) !important;
}

section[data-testid="stSidebar"] .block-container{
  padding-top: 1rem !important;
  background: var(--sb-bg) !important;
}

button[data-testid="collapsedControl"]{ display:none !important; }
section[data-testid="stSidebar"] div[data-testid="stSidebarNav"]{ display:none; }

.sb-app-title{
  font-size: 1.35rem;
  font-weight: 850;
  color: var(--text);
  line-height: 1.2;
  text-align: center;
  margin: .35rem 0 .55rem 0;
}
.sb-divider{
  height: 3px;
  background: #D1D5DB;
  border-radius: 999px;
  margin: .4rem 0 1rem 0;
}
.sb-h1{
  font-size: 1.05rem;
  font-weight: 800;
  color: var(--text);
  margin: 0 0 .35rem 0;
}
.sb-sub{
  font-size: .86rem;
  color: var(--muted);
  margin: 0 0 .85rem 0;
}

/* section look */
.sb-card{
  border: 1.5px solid var(--border);
  background: var(--card-bg) !important;
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
}
.sb-sec-title{
  font-size: .95rem;
  font-weight: 750;
  color: var(--text);
  margin: 0 0 2px 0;
}
.sb-sec-sub{
  font-size: .82rem;
  color: var(--muted);
  margin: 0 0 10px 0;
}
.sb-mini-sep{
  height: 1px;
  background: #E5E7EB;
  margin: 10px 2px;
}

/* uploader dropzone */
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"]{
  background: var(--card-bg) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
}

/* Buttons */
section[data-testid="stSidebar"] .stButton button{
  border-radius: 8px !important;
}

/* Main area column cleanup */
div[data-testid="column"] > div:first-child{
  margin-top: 0 !important;
  padding-top: 0 !important;
}
</style>
""",
    unsafe_allow_html=True
)

# -------------------- JS: remove pills + force sidebar open --------------------
components.html(
    """
<script>
(function () {
  function isPill(el) {
    if (!el || el.nodeType !== 1) return false;
    const cs = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    const br = parseFloat(cs.borderRadius || "0");
    const bw = parseFloat(cs.borderTopWidth || "0");
    const hasBorder = bw >= 1 && (cs.borderStyle || "").includes("solid");
    const h = rect.height;
    const w = rect.width;
    const sizeLike = (h >= 26 && h <= 52) && (w >= 180);
    const roundLike = br >= 12;
    const txt = (el.innerText || "").trim();
    const emptyLike = txt.length === 0;
    return sizeLike && roundLike && hasBorder && emptyLike;
  }

  function hidePills() {
    const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
    if (!sidebar) return;
    const candidates = sidebar.querySelectorAll("div, section, label");
    candidates.forEach(el => {
      if (isPill(el)) {
        const container =
          el.closest('div[data-testid="stElementContainer"]') ||
          el.closest('div[data-testid="stVerticalBlock"]') ||
          el;
        container.style.display = "none";
        container.style.height = "0px";
        container.style.margin = "0";
        container.style.padding = "0";
        container.style.border = "0";
        container.style.boxShadow = "none";
      }
    });
  }

  // --- Force sidebar to stay expanded ---
  function forceSidebarOpen() {
    const doc = window.parent.document;


    const toggle =
      doc.querySelector('button[aria-label="Toggle sidebar"]') ||
      doc.querySelector('button[title="Toggle sidebar"]') ||
      doc.querySelector('button[kind="header"]') ||  // fallback (some builds)
      null;

    if (!toggle) return;

    // If it's collapsed, aria-expanded becomes "false"
    const expanded = toggle.getAttribute("aria-expanded");
    if (expanded === "false") {
      toggle.click(); // reopen
    }
  }

  function runAll(){
    hidePills();
    forceSidebarOpen();
  }

  runAll();

  // Observe sidebar + header (because toggle lives in header in some builds)
  const doc = window.parent.document;
  const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
  const header = doc.querySelector('header');

  const obs = new MutationObserver(() => runAll());
  if (sidebar) obs.observe(sidebar, { childList: true, subtree: true });
  if (header) obs.observe(header, { childList: true, subtree: true });

  // retry for delayed hydration
  let tries = 0;
  const iv = setInterval(() => {
    runAll();
    tries++;
    if (tries > 80) clearInterval(iv); // ~8s
  }, 100);
})();
</script>
""",
    height=0,
    width=0,
)

if Path(BANNER).exists():
    st.image(BANNER, use_container_width=True)

st.markdown("## Introduction")

st.markdown("""
The **AI-Powered Apple Leaf Specialist** is a lightweight, real-time computer vision application designed to help apple growers quickly identify common apple leaf conditions using a single photo. Using a fine-tuned ResNet-18 deep learning model with calibrated probability outputs, the system classifies leaf images into one of four conditions: **healthy, scab, rust, or black rot**. When the model confidence is low, the app conservatively routes the result to an **“unknown”** label to avoid overconfident misclassification.

The app supports both image upload and live camera capture, applies deterministic preprocessing, and runs local TorchScript inference on CPU without requiring cloud connectivity. After prediction, the app displays class probabilities and provides **tailored care recommendations specific to the predicted disease**, helping users take the next best step (prevention, treatment, and best practices) based on the detected condition.
""")

st.markdown("---")

# -------------------- Helpers --------------------
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

def compute_brightness(pil_img: Image.Image) -> float:
    arr = np.asarray(pil_img.resize((256, 256))).astype(np.float32) / 255.0
    y = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
    return float(y.mean())

def green_coverage_soft(pil_img: Image.Image) -> float:
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

def _pretty(lab: str) -> str:
    return lab.replace("_", " ").title()

def render_prob_bars_native(prob_map: dict):
    st.markdown("**Apple Disease Probability**")
    order = ["black_rot", "healthy", "scab", "rust"]
    for lab in order:
        p = float(prob_map.get(lab, 0.0))
        c1, c2, c3 = st.columns([1.6, 6, 1.2])
        with c1: st.write(_pretty(lab))
        with c2:
            try:
                st.progress(p)
            except Exception:
                st.progress(int(p * 100))
        with c3: st.write(f"{p*100:.1f}%")

# -------------------- Posters --------------------
CARE_POSTERS = {
    "black_rot": "black_rot_care_v2.jpg",
    "healthy":   "healthy_care_v2.jpg",
    "scab":      "scab_care_v2.jpg",
    "rust":      "rust_care_v2.jpg",
}

# -------------------- Session state --------------------
if "show_camera" not in st.session_state:     st.session_state.show_camera = False
if "source" not in st.session_state:         st.session_state.source = None
if "captured" not in st.session_state:       st.session_state.captured = None
if "upload" not in st.session_state:         st.session_state.upload = None
if "keep_camera_on" not in st.session_state: st.session_state.keep_camera_on = False

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

# -------------------- Sidebar UI --------------------
with st.sidebar:
    if Path(APP_LOGO).exists():
        b64 = base64.b64encode(Path(APP_LOGO).read_bytes()).decode()
        ext = Path(APP_LOGO).suffix.lstrip(".").lower() or "png"
        st.markdown(
            f"""
            <div style="display:flex;justify-content:center;margin-top:-2rem;">
              <img src="data:image/{ext};base64,{b64}" style="max-width:220px;height:auto;" />
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div style="text-align:center;font-size:52px">🍎</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-app-title">AI-Powered Apple Leaf Specialist</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-h1">Add a leaf photo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Upload a file or take a photo using your camera.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-card">', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec-title">Upload Photo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec-sub">Drop a JPG/PNG here, or browse</div>', unsafe_allow_html=True)
    st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader", on_change=on_upload_change)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-mini-sep"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-card">', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec-title">Record Photo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec-sub">Use your device camera</div>', unsafe_allow_html=True)

    if not st.session_state.show_camera:
        c_hint, c_btn = st.columns([6, 4], vertical_alignment="center")
        with c_hint:
            st.caption('Tap "Open camera" to take a photo.')
        with c_btn:
            if st.button("Open camera", key="open_cam_btn"):
                open_camera()
    else:
        cap = st.camera_input("", key="camera_input")
        if cap is not None:
            st.session_state.captured = cap
            st.session_state.source = "camera"
            if not st.session_state.keep_camera_on:
                st.session_state.show_camera = False
        st.button("Close camera", on_click=close_camera, key="close_cam_btn")

    #st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")
    st.write("""
    ### Contacts
    [![](https://img.shields.io/badge/GitHub-Follow-informational)](https://github.com/akthammomani)
    [![](https://img.shields.io/badge/Linkedin-Connect-informational)](https://www.linkedin.com/in/akthammomani/)
    [![](https://img.shields.io/badge/Open-Issue-informational)](https://github.com/akthammomani/ai_powered_apple_leaf_specialist/issues)
    ###### © Aktham Momani, 2025. All rights reserved.
    """)

# -------------------- Active source --------------------
file = st.session_state.captured if st.session_state.source == "camera" else (
    st.session_state.upload if st.session_state.source == "upload" else None
)

# -------------------- Main inference path --------------------
if file:
    pil = load_pil(file)

    b = compute_brightness(pil)
    if b < dark_thr:
        st.warning(f"Image appears too dark (brightness {b:.2f}). Retake under brighter, even lighting.")
        st.stop()
    if b > bright_thr:
        st.warning(f"Image appears too bright/washed-out (brightness {b:.2f}). Retake avoiding direct glare.")
        st.stop()

    if st.session_state.source == "camera":
        bypass_gate = st.checkbox("Bypass leaf check for this camera image", value=False)
        ok_leaf, cov, tex = is_leaf_like(pil, cov_min=cov_min, cov_max=0.98, tex_min=tex_min)
        if not (ok_leaf or bypass_gate):
            st.warning(
                f"This photo might not be a single leaf (green_coverage≈{cov:.2f}, texture≈{tex:.0f}). "
                "Retake: fill the frame with one leaf in even lighting, sharp focus — or tick the bypass to proceed."
            )
            st.stop()

    probs = predict_probs(pil)
    pred_label, pred_conf, _ = decide(probs, labels, THRESHOLD)
    prob_map = {lab: float(probs[i]) for i, lab in enumerate(labels)}

    r1_left, med, r1_right = st.columns([0.5, 0.5, 1], gap="large")
    with r1_left:
        st.markdown("### Your Image:")
        st.image(ImageOps.contain(pil, (PREVIEW_MAX_W, PREVIEW_MAX_H)), use_container_width=False)

    with med:
        st.markdown("### Learn More")
        st.markdown("[![](https://img.shields.io/badge/GitHub%20-AI--Powered%20Apple%20Leaf%20Specialist-informational)](https://github.com/akthammomani/ai_powered_apple_leaf_specialist)")

    with r1_right:
        st.markdown("### Predicted Apple Disease Label is:")
        st.markdown(f"**{_pretty(pred_label)}** with **{pred_conf*100:.0f}%** Confidence")
        render_prob_bars_native(prob_map)
        st.write("#### Learn More")
        st.markdown("[![](https://img.shields.io/badge/GitHub-Model%20Notebook-informational)](https://github.com/akthammomani/ai_powered_apple_leaf_specialist/blob/main/Notebooks/Modeling_AI_Powered_Apple_Leaf_Specialist.ipynb)")


    st.markdown(f"### Apple – {_pretty(pred_label)} Care Recommendations:")
    poster_path = CARE_POSTERS.get(pred_label, CARE_POSTERS["healthy"])
    if not Path(poster_path).exists():
        st.info("Care poster not found. Please add the JPGs next to app.py.")
    else:
        st.image(poster_path, use_container_width=True)
        st.write("""
        ###### ***Disclaimer***
        *This app is not a substitute for professional agricultural advice, diagnosis, or treatment. Field conditions, pests, and diseases can vary widely. Always consult a qualified agronomist, crop advisor, or local extension service before making decisions that could affect tree health, spray plans, or harvest.*
        """)
else:
    st.info("Use the sidebar to upload a photo or open the camera to begin.")
