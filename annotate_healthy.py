"""
Streamlit app for annotating healthy artery bounding boxes.
Draw rectangle by dragging the mouse on the image canvas.

Usage:
    conda activate stenosis
    streamlit run annotate_healthy.py
"""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import streamlit.components.v1 as components

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
IMAGES_DIR = BASE / "data" / "stenosis_arcade" / "train" / "images"
LABELS_DIR = BASE / "data" / "stenosis_arcade" / "train" / "labels"
HEALTHY_DIR = BASE / "data" / "stenosis_arcade" / "train" / "labels_healthy"
HEALTHY_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_DIR = BASE / "rect_component"

COLORS = {0: (0, 255, 0), 1: (255, 50, 50)}
CNAMES = {0: "stenosis_0", 1: "stenosis_1"}
HEALTHY_CLS = 0
DISP_W = 800

# Register custom bidirectional component
_rect_drawer = components.declare_component("rect_drawer", path=str(COMPONENT_DIR))


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
@st.cache_data
def image_stems():
    return sorted([p.stem for p in IMAGES_DIR.glob("*.png")], key=int)


def load_boxes(stem):
    p = LABELS_DIR / f"{stem}.txt"
    out = []
    if p.exists():
        for ln in p.read_text().strip().splitlines():
            ps = ln.split()
            if len(ps) >= 5:
                out.append((int(ps[0]), *map(float, ps[1:5])))
    return out


def has_healthy(stem):
    return (HEALTHY_DIR / f"{stem}.txt").exists()


def load_healthy(stem):
    p = HEALTHY_DIR / f"{stem}.txt"
    if p.exists():
        t = p.read_text().strip()
        if t:
            ps = t.split()
            if len(ps) >= 5:
                return (int(ps[0]), *map(float, ps[1:5]))
    return None


def save_healthy(stem, cx, cy, w, h):
    (HEALTHY_DIR / f"{stem}.txt").write_text(
        f"{HEALTHY_CLS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def del_healthy(stem):
    p = HEALTHY_DIR / f"{stem}.txt"
    if p.exists():
        p.unlink()


def yolo2px(cx, cy, bw, bh, iw, ih):
    return ((cx - bw / 2) * iw, (cy - bh / 2) * ih,
            (cx + bw / 2) * iw, (cy + bh / 2) * ih)


def n_annotated(stems):
    return sum(1 for s in stems if has_healthy(s))


def img_to_b64(stem):
    """Load image, resize to DISP_W, return base64 + display height."""
    img = Image.open(IMAGES_DIR / f"{stem}.png").convert("RGB")
    ow, oh = img.size
    sc = DISP_W / ow
    dh = int(oh * sc)
    img = img.resize((DISP_W, dh), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), dh


def stenosis_boxes_px(stem, dh):
    """Convert YOLO stenosis boxes to pixel coords on display image."""
    out = []
    for cls, cx, cy, bw, bh in load_boxes(stem):
        x1, y1, x2, y2 = yolo2px(cx, cy, bw, bh, DISP_W, dh)
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls})
    return out


def healthy_box_px(stem, dh):
    """Convert saved healthy box to pixel coords."""
    h = load_healthy(stem)
    if h:
        _, cx, cy, bw, bh = h
        x1, y1, x2, y2 = yolo2px(cx, cy, bw, bh, DISP_W, dh)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return None


# ──────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Healthy Annotator", layout="wide")
st.title("🫀 Healthy Artery Annotator")

stems = image_stems()
total = len(stems)

for k, v in [("idx", 0), ("auto_adv", True)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Controls")
    show_st = st.checkbox("Show stenosis boxes", True)
    st.session_state.auto_adv = st.checkbox("Auto-advance after save",
                                             st.session_state.auto_adv)
    skip_done = st.checkbox("Skip annotated", False)
    st.divider()

    flt = [s for s in stems if not has_healthy(s)] if skip_done else list(stems)
    if not flt:
        st.success("🎉 All 1000 images annotated!")
        st.stop()

    ann = n_annotated(stems)
    st.metric("Progress", f"{ann}/{total}")
    st.progress(ann / total if total else 0)
    st.divider()

    st.session_state.idx = max(0, min(st.session_state.idx, len(flt) - 1))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅ Prev", use_container_width=True):
            st.session_state.idx = max(0, st.session_state.idx - 1)
            st.rerun()
    with c2:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.idx = min(len(flt) - 1, st.session_state.idx + 1)
            st.rerun()

    ni = st.number_input("Go to #", 1, len(flt), st.session_state.idx + 1) - 1
    if ni != st.session_state.idx:
        st.session_state.idx = ni
        st.rerun()

    st.divider()
    if st.button("⏭ Next unannotated", use_container_width=True):
        for i, s in enumerate(flt):
            if not has_healthy(s):
                st.session_state.idx = i
                st.rerun()

# ── Main area ──
stem = flt[st.session_state.idx]
existing = load_healthy(stem)
boxes = load_boxes(stem)

icon = "✅" if existing else "⬜"
st.markdown(f"### {icon} **{stem}.png**  ({st.session_state.idx + 1}/{len(flt)})  "
            f"| Stenosis: {len(boxes)}")

if existing:
    st.info(f"Saved healthy box: cx={existing[1]:.4f}  cy={existing[2]:.4f}  "
            f"w={existing[3]:.4f}  h={existing[4]:.4f}")

st.caption("🖱️ **Drag** on the image to draw a healthy artery bounding box. "
           "It auto-saves on mouse release.")

# Prepare data for the canvas component
b64, dh = img_to_b64(stem)
st_boxes = stenosis_boxes_px(stem, dh) if show_st else []
h_box = healthy_box_px(stem, dh)

# Render the interactive canvas component
canvas_key = f"rc_{stem}_{st.session_state.idx}"
result = _rect_drawer(
    image_b64=b64,
    width=DISP_W,
    height=dh,
    stenosis_boxes=st_boxes,
    healthy_box=h_box,
    key=canvas_key,
    default=None,
)

# Process the drawn rectangle
if result is not None and isinstance(result, dict):
    x1, y1 = result["x1"], result["y1"]
    x2, y2 = result["x2"], result["y2"]
    # Convert pixel → normalized YOLO
    cx = ((x1 + x2) / 2) / DISP_W
    cy = ((y1 + y2) / 2) / dh
    bw = (x2 - x1) / DISP_W
    bh = (y2 - y1) / dh
    # Make square (use max side)
    side = max(bw, bh)
    save_healthy(stem, cx, cy, side, side)
    st.toast(f"✅ Saved healthy box for {stem}.png!")
    if st.session_state.auto_adv:
        st.session_state.idx = min(len(flt) - 1, st.session_state.idx + 1)
    st.rerun()

# Delete button
if st.button("🗑️ Delete annotation", use_container_width=False):
    del_healthy(stem)
    st.rerun()

st.divider()
st.caption(f"Done: {n_annotated(stems)}/{total}")
