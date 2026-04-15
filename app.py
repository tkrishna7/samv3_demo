"""
SAM2 Segment & Style — local replication of Meta's Segment Anything demo.

Features:
  • Click any object to segment it with SAM2
  • Background Blur  — Gaussian blur behind the selected object
  • Background Color Fill — solid colour behind the selected object
  • Object Highlight — semi-transparent colour overlay on the object itself

Run:
    uv run streamlit run app.py
"""

import io
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# ── paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path("weights")
MODEL_CONFIGS = {
    "small (46 MB)": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "tiny (38 MB)": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
    "base+ (80 MB)": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
    "large (224 MB)": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
}

# ── device helpers ─────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── model loading (cached so it only runs once per session) ────────────────────
@st.cache_resource(show_spinner="Loading SAM2 model…")
def load_predictor(cfg: str, ckpt: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(cfg, ckpt, device=device)
    return SAM2ImagePredictor(model)


# ── effect helpers ─────────────────────────────────────────────────────────────
def apply_blur_bg(image_np: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Blur background, keep foreground sharp."""
    if radius % 2 == 0:
        radius += 1
    blurred = cv2.GaussianBlur(image_np, (radius, radius), 0)
    result = image_np.copy()
    result[mask == 0] = blurred[mask == 0]
    return result


def apply_color_fill_bg(
    image_np: np.ndarray, mask: np.ndarray, color: tuple
) -> np.ndarray:
    """Replace background with a solid colour."""
    result = image_np.copy()
    result[mask == 0] = color
    return result


def apply_highlight_fg(
    image_np: np.ndarray, mask: np.ndarray, color: tuple, alpha: float
) -> np.ndarray:
    """Semi-transparent colour overlay on the foreground object."""
    result = image_np.copy().astype(np.float32)
    c = np.array(color, dtype=np.float32)
    fg = mask == 1
    result[fg] = result[fg] * (1 - alpha) + c * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_point_on_image(image_np: np.ndarray, x: int, y: int) -> np.ndarray:
    """Draw a small crosshair to show the selected point."""
    annotated = image_np.copy()
    r = 8
    cv2.circle(annotated, (x, y), r, (255, 255, 255), 3)
    cv2.circle(annotated, (x, y), r, (50, 200, 50), 2)
    cv2.line(annotated, (x - r - 4, y), (x + r + 4, y), (50, 200, 50), 2)
    cv2.line(annotated, (x, y - r - 4), (x, y + r + 4), (50, 200, 50), 2)
    return annotated


def mask_to_rgba_overlay(mask: np.ndarray, color=(50, 200, 50), alpha=0.4) -> np.ndarray:
    """Create a transparent RGBA overlay showing the mask boundary."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask == 1, :3] = color
    overlay[mask == 1, 3] = int(255 * alpha)
    return overlay


# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segment & Style — SAM2",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS (matches Meta demo's clean aesthetic) ──────────────────────────
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      .stButton > button {
          width: 100%;
          border-radius: 8px;
          font-weight: 600;
      }
      div[data-testid="stImage"] img { border-radius: 10px; }
      .effect-label {
          font-size: 0.8rem;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: 4px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✂️  Segment & Style")
    st.caption("Powered by Meta SAM2")
    st.divider()

    # -- Image source
    st.subheader("Image")
    source = st.radio(
        "Source", ["Use example image", "Upload your own"], label_visibility="collapsed"
    )
    uploaded = None
    if source == "Upload your own":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

    st.divider()

    # -- Model selection
    st.subheader("Model")
    available_models = {
        k: v
        for k, v in MODEL_CONFIGS.items()
        if (WEIGHTS_DIR / v[1]).exists()
    }
    if not available_models:
        st.error(
            "No weights found in `weights/`.\n\n"
            "Run:  `uv run python download_weights.py`"
        )
        st.stop()

    model_choice = st.selectbox(
        "Checkpoint",
        list(available_models.keys()),
        label_visibility="collapsed",
    )
    cfg_name, ckpt_name = available_models[model_choice]
    ckpt_path = str(WEIGHTS_DIR / ckpt_name)
    device = get_device()
    st.caption(f"Device: **{device.upper()}**")

    st.divider()

    # -- Effects panel
    st.subheader("Add effects")

    st.markdown('<div class="effect-label">Object effects</div>', unsafe_allow_html=True)
    enable_highlight = st.toggle("Highlight object", value=False)
    if enable_highlight:
        hl_col1, hl_col2 = st.columns([2, 1])
        with hl_col1:
            highlight_color_hex = st.color_picker("Colour", "#33C87E", key="hl_color")
        with hl_col2:
            highlight_alpha = st.slider("Opacity", 0.1, 0.9, 0.45, 0.05, key="hl_alpha")

    st.markdown('<div class="effect-label" style="margin-top:12px">Background effects</div>', unsafe_allow_html=True)
    effect_bg = st.radio(
        "Background effect",
        ["None", "Blur", "Color Fill"],
        label_visibility="collapsed",
    )

    if effect_bg == "Blur":
        blur_radius = st.slider("Blur strength", 5, 101, 35, 2, key="blur_rad")
    elif effect_bg == "Color Fill":
        fill_color_hex = st.color_picker("Fill colour", "#1A1A2E", key="fill_color")

    st.divider()
    if st.button("Clear selection", use_container_width=True):
        for k in ["click_x", "click_y", "mask", "result_image"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── load image ─────────────────────────────────────────────────────────────────
if source == "Use example image":
    example_path = Path("example_shot.png")
    if not example_path.exists():
        st.error("example_shot.png not found in the project directory.")
        st.stop()
    pil_image = Image.open(example_path).convert("RGB")
else:
    if uploaded is None:
        st.info("Upload an image from the sidebar to get started.")
        st.stop()
    pil_image = Image.open(uploaded).convert("RGB")

image_np = np.array(pil_image)
orig_h, orig_w = image_np.shape[:2]

# ── load model ─────────────────────────────────────────────────────────────────
predictor = load_predictor(cfg_name, ckpt_path, device)

# ── main layout ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

DISPLAY_WIDTH = 680  # px width for the interactive image panel

with col_left:
    st.subheader("1 — Click on the object to segment")
    st.caption("Click anywhere on the image to place a point prompt. SAM2 will segment the object at that point.")

    # Determine what to show in the interactive panel
    if "click_x" in st.session_state and "mask" in st.session_state:
        # Show annotated image (point drawn on original)
        annotated = draw_point_on_image(
            image_np, st.session_state["click_x"], st.session_state["click_y"]
        )
        display_pil = Image.fromarray(annotated)
    else:
        display_pil = pil_image

    coords = streamlit_image_coordinates(display_pil, key="img_click", width=DISPLAY_WIDTH)

    if coords is not None:
        # streamlit-image-coordinates returns coords in display-image space
        scale_x = orig_w / DISPLAY_WIDTH
        scale_y = orig_h / (DISPLAY_WIDTH * orig_h / orig_w)
        raw_x = int(coords["x"] * scale_x)
        raw_y = int(coords["y"] * scale_y)
        raw_x = max(0, min(raw_x, orig_w - 1))
        raw_y = max(0, min(raw_y, orig_h - 1))

        if (
            st.session_state.get("click_x") != raw_x
            or st.session_state.get("click_y") != raw_y
        ):
            st.session_state["click_x"] = raw_x
            st.session_state["click_y"] = raw_y

            with st.spinner("Segmenting…"):
                predictor.set_image(image_np)
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[raw_x, raw_y]]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                st.session_state["mask"] = masks[best].astype(np.uint8)
            st.rerun()

    if "click_x" in st.session_state:
        x, y = st.session_state["click_x"], st.session_state["click_y"]
        st.caption(f"Selected point: ({x}, {y}) — original image coordinates")

with col_right:
    st.subheader("2 — Preview with effects")

    if "mask" not in st.session_state:
        st.info("Click an object on the left to begin.")
    else:
        mask = st.session_state["mask"]
        result = image_np.copy()

        # Apply background effect first
        if effect_bg == "Blur":
            result = apply_blur_bg(result, mask, blur_radius)
        elif effect_bg == "Color Fill":
            result = apply_color_fill_bg(result, mask, hex_to_rgb(fill_color_hex))

        # Apply object highlight on top
        if enable_highlight:
            result = apply_highlight_fg(
                result, mask, hex_to_rgb(highlight_color_hex), highlight_alpha
            )

        result_pil = Image.fromarray(result)
        st.image(result_pil, use_container_width=True)

        # Download button
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            "Download result",
            data=buf.getvalue(),
            file_name="segmented_result.png",
            mime="image/png",
            use_container_width=True,
        )

        # Mask visualisation
        with st.expander("Show segmentation mask"):
            mask_vis = (mask * 255).astype(np.uint8)
            st.image(mask_vis, caption="Binary mask (white = selected object)", use_container_width=True)

# ── footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with [Meta SAM2](https://github.com/facebookresearch/sam2) · "
    "Replicating [Meta AI Demos](https://aidemos.meta.com/segment-anything)"
)
