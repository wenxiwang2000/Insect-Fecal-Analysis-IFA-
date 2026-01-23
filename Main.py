"""
TURD-like spot detector — organized 3-panel layout + bottom save/export bar.

Features preserved from the artistic layout:
- Left: image upload + ROI draw (rect/circle) + optional session JSON load + "use saved ROI"
- Middle: parameters (incl. sensitivity) + advanced controls + metric notes
- Right: ROI crop + annotated overlay + masks (binary/ROI/pink/arena) + quick summary
- Bottom: export CSV/PNG + save session JSON (reload later)

Requires:
  pip install streamlit opencv-python numpy pandas pillow
  pip install streamlit-drawable-canvas-fix

Run:
  streamlit run turd_like_spot_ui_layout_organized.py
"""

from __future__ import annotations

import json
import math
from io import BytesIO
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas  # streamlit-drawable-canvas-fix


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class ROI:
    kind: str  # "rect" or "circle"
    x0: int
    y0: int
    x1: int
    y1: int
    cx: Optional[int] = None
    cy: Optional[int] = None
    r: Optional[int] = None

    def clamp(self, w: int, h: int) -> "ROI":
        """Clamp ROI to image bounds. Keeps kind semantics."""
        x0 = int(max(0, min(self.x0, w - 1)))
        x1 = int(max(1, min(self.x1, w)))
        y0 = int(max(0, min(self.y0, h - 1)))
        y1 = int(max(1, min(self.y1, h)))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)

        out = ROI(self.kind, x0, y0, x1, y1, self.cx, self.cy, self.r)

        if out.kind == "circle" and out.cx is not None and out.cy is not None and out.r is not None:
            # Clamp the circle center to the image, and keep a valid bounding box for cropping.
            cx = int(max(0, min(out.cx, w - 1)))
            cy = int(max(0, min(out.cy, h - 1)))
            r = int(max(1, out.r))

            x0 = int(max(0, min(cx - r, w - 1)))
            y0 = int(max(0, min(cy - r, h - 1)))
            x1 = int(max(x0 + 1, min(cx + r, w)))
            y1 = int(max(y0 + 1, min(cy + r, h)))

            out.cx, out.cy, out.r = cx, cy, r
            out.x0, out.y0, out.x1, out.y1 = x0, y0, x1, y1
        return out


def white_candidate_mask_hsv(bgr: np.ndarray, s_max: int = 40, v_min: int = 220) -> np.ndarray:
    """Fast 'white-ish' pixel mask used for arena heuristics (HSV-based)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    return (((s <= s_max) & (v >= v_min)).astype(np.uint8) * 255)


@dataclass
class Params:
    # Core detection
    min_area: int = 10
    max_area: int = 4000
    sensitivity: int = 200
    n_flies: int = 8

    # Toggles
    remove_pink_background: bool = True
    use_arenas: bool = True

    # Advanced
    rod_circularity_cutoff: float = 0.60
    morph_open: int = 3
    morph_close: int = 7

    # Pink mask (HSV)
    pink_h_low: int = 145
    pink_h_high: int = 179
    pink_s_min: int = 40
    pink_v_min: int = 0
    pink_close: int = 9

    # Arena mask (white circles)
    arena_s_max: int = 40
    arena_v_min: int = 220
    arena_close: int = 15
    arena_erode: int = 9
    arena_min_area: int = 20000
    arena_max_area: int = 400000

    # Derived (computed from sensitivity)
    adaptive_block_size: int = 31
    adaptive_offset_c: int = 10


# =============================================================================
# Image helpers
# =============================================================================
def resize_for_display(img: Image.Image, max_w: int = 1100) -> Tuple[Image.Image, float]:
    w, _h = img.size
    if w <= max_w:
        return img, 1.0
    scale = max_w / float(w)
    return img.resize((int(w * scale), int(img.size[1] * scale))), scale


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def apply_mask_alpha(pil_img: Image.Image, mask_u8: np.ndarray) -> Image.Image:
    """Return an RGBA image where alpha channel = mask (0..255)."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    rgba = np.array(pil_img.convert("RGBA"))
    rgba[:, :, 3] = mask_u8
    return Image.fromarray(rgba, mode="RGBA")


# =============================================================================
# ROI from canvas
# =============================================================================
def _get_last_shape(canvas_json: Dict, kind: str) -> Optional[Dict]:
    if not canvas_json or "objects" not in canvas_json:
        return None
    shapes = [o for o in canvas_json["objects"] if o.get("type") == kind]
    return shapes[-1] if shapes else None


def roi_from_canvas(canvas_json: Dict, draw_mode: str, scale: float) -> Optional[ROI]:
    shape = _get_last_shape(canvas_json, "rect" if draw_mode == "rect" else "circle")
    if not shape:
        return None

    left = float(shape.get("left", 0))
    top = float(shape.get("top", 0))
    sx = float(shape.get("scaleX", 1.0))
    sy = float(shape.get("scaleY", 1.0))

    if draw_mode == "rect":
        w = float(shape.get("width", 0)) * sx
        h = float(shape.get("height", 0)) * sy
        x0 = int(left / scale)
        y0 = int(top / scale)
        x1 = int((left + w) / scale)
        y1 = int((top + h) / scale)
        return ROI("rect", x0, y0, x1, y1)

    # circle
    r = float(shape.get("radius", 0))
    s = (sx + sy) / 2.0  # keep circular
    rr = r * s
    cx = (left + rr) / scale
    cy = (top + rr) / scale
    r0 = rr / scale

    cx_i = int(round(cx))
    cy_i = int(round(cy))
    r_i = int(round(r0))
    return ROI("circle", cx_i - r_i, cy_i - r_i, cx_i + r_i, cy_i + r_i, cx=cx_i, cy=cy_i, r=r_i)


def make_roi_mask(w: int, h: int, roi: ROI) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if roi.kind == "rect":
        m[roi.y0:roi.y1, roi.x0:roi.x1] = 255
    else:
        assert roi.cx is not None and roi.cy is not None and roi.r is not None
        cv2.circle(m, (roi.cx, roi.cy), roi.r, 255, thickness=-1)
    return m


# =============================================================================
# Vision pipeline
# =============================================================================
def adaptive_threshold(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, int(c)
    )


def morphology(bw: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    out = bw
    if open_k > 1:
        k = np.ones((open_k, open_k), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    if close_k > 1:
        k = np.ones((close_k, close_k), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return out


def circularity(area: float, peri: float) -> float:
    if peri <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (peri * peri))


def pink_background_mask_hsv(
    bgr: np.ndarray,
    h_low: int = 145,
    h_high: int = 179,
    s_min: int = 40,
    v_min: int = 0,
    close_k: int = 9,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if h_low <= h_high:
        in_h = (h >= h_low) & (h <= h_high)
    else:
        in_h = (h >= h_low) | (h <= h_high)

    pink = in_h & (s >= s_min) & (v >= v_min)
    m = (pink.astype(np.uint8) * 255)
    if close_k and close_k > 1:
        k = np.ones((close_k, close_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m


def detect_arenas_white_regions(
    bgr: np.ndarray,
    s_max: int = 40,
    v_min: int = 220,
    close_k: int = 15,
    erode_k: int = 9,
    min_area: int = 20000,
    max_area: int = 400000,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s <= s_max) & (v >= v_min)).astype(np.uint8) * 255

    if close_k > 1:
        k = np.ones((close_k, close_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        cv2.drawContours(out, [c], -1, 255, thickness=-1)

    if erode_k and erode_k > 1:
        k = np.ones((erode_k, erode_k), np.uint8)
        out = cv2.erode(out, k, iterations=1)

    return out


def measure_spots_turd_style(
    bgr: np.ndarray,
    bw: np.ndarray,
    min_area: int,
    max_area: int,
    rod_circ_cutoff: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rows: List[Dict] = []
    spot_union = np.zeros(bw.shape, dtype=np.uint8)

    did = 0
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < min_area or a > max_area:
            continue
        p = float(cv2.arcLength(c, True))
        circ = circularity(a, p)

        m = np.zeros(bw.shape, dtype=np.uint8)
        cv2.drawContours(m, [c], -1, 255, thickness=-1)
        spot_union = cv2.bitwise_or(spot_union, m)

        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = float("nan"), float("nan")

        mean_b, mean_g, mean_r, _ = cv2.mean(bgr, mask=m)
        mean_h, mean_l, mean_s, _ = cv2.mean(hls, mask=m)

        mean_h_deg = (mean_h * 2.0) % 360.0
        mean_l_n = mean_l / 255.0
        mean_s_n = mean_s / 255.0

        # TURD-like IOD: area × (1 - MeanL)
        iod = a * (1.0 - mean_l_n)
        rod = 1 if circ < rod_circ_cutoff else 0

        did += 1
        rows.append(
            {
                "DepositID": did,
                "X": cx,
                "Y": cy,
                "Area_px": a,
                "Perimeter_px": p,
                "Circularity": circ,
                "ROD": rod,
                "IOD": iod,
                "MeanR": mean_r / 255.0,
                "MeanG": mean_g / 255.0,
                "MeanB": mean_b / 255.0,
                "MeanH_deg": mean_h_deg,
                "MeanL": mean_l_n,
                "MeanS": mean_s_n,
            }
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "DepositID",
            "X",
            "Y",
            "Area_px",
            "Perimeter_px",
            "Circularity",
            "ROD",
            "IOD",
            "MeanR",
            "MeanG",
            "MeanB",
            "MeanH_deg",
            "MeanL",
            "MeanS",
        ]
    )
    return df, spot_union


def plate_summary(df: pd.DataFrame, n_flies: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "N_spots": 0,
                    "TotalArea_px": 0.0,
                    "TotalIOD": 0.0,
                    "N_flies": int(n_flies),
                    "Spots_per_fly": np.nan,
                    "Area_per_fly": np.nan,
                    "IOD_per_fly": np.nan,
                }
            ]
        )

    n = int(df.shape[0])
    total_area = float(df["Area_px"].sum())
    total_iod = float(df["IOD"].sum())

    out = {"N_spots": n, "TotalArea_px": total_area, "TotalIOD": total_iod, "N_flies": int(n_flies)}
    if n_flies and n_flies > 0:
        out["Spots_per_fly"] = n / float(n_flies)
        out["Area_per_fly"] = total_area / float(n_flies)
        out["IOD_per_fly"] = total_iod / float(n_flies)
    else:
        out["Spots_per_fly"] = np.nan
        out["Area_per_fly"] = np.nan
        out["IOD_per_fly"] = np.nan

    return pd.DataFrame([out])


def annotate(bgr: np.ndarray, bw: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    out = bgr.copy()
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    for _, r in df.iterrows():
        if np.isnan(r["X"]) or np.isnan(r["Y"]):
            continue
        cv2.putText(
            out,
            str(int(r["DepositID"])),
            (int(r["X"]), int(r["Y"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return out


# =============================================================================
# UI styling
# =============================================================================
CSS = """
<style>
    .panel {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 18px;
        padding: 14px 16px 10px 16px;
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        margin-bottom: 14px;
    }
    .panel h3 {
        margin-top: 0px;
        margin-bottom: 10px;
        font-weight: 650;
        letter-spacing: 0.2px;
    }
    .tiny {
        color: rgba(0,0,0,0.55);
        font-size: 12px;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .bar {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 10px 14px;
        background: rgba(255,255,255,0.85);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }
</style>
"""


def panel_open(title: str) -> None:
    st.markdown(f'<div class="panel"><h3>{title}</h3>', unsafe_allow_html=True)


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Session + params helpers
# =============================================================================
def load_session_json(file) -> Dict:
    if file is None:
        return {}
    try:
        return json.loads(file.read().decode("utf-8"))
    except Exception:
        return {}


def params_from_session(session: Dict) -> Params:
    """Build Params from session dict (if any), otherwise defaults."""
    p = Params()
    # Optional: allow both old keys and new keys; ignore unknown keys.
    for k, v in (session or {}).items():
        if hasattr(p, k):
            try:
                setattr(p, k, v)
            except Exception:
                pass
    return p


def derive_threshold_params(p: Params) -> Params:
    """Map sensitivity -> adaptive threshold parameters and store them on Params.

    Original UI used a 0–100 slider. We now allow *unbounded* sensitivity values,
    but we keep the same behavior for 0–100 and extend safely beyond 100 without
    producing invalid (negative/zero) adaptive-threshold parameters.
    """
    s = max(0.0, float(p.sensitivity))

    # Preserve original linear mapping for 0..100
    if s <= 100.0:
        block_size = float(np.interp(s, [0, 100], [71, 15]))
        offset_c = float(np.interp(s, [0, 100], [18, 4]))
    else:
        # For values >100, asymptotically approach tighter settings (never invalid).
        extra = s - 100.0
        # block_size: 15 -> 3 as sensitivity grows
        block_size = 15.0 - (15.0 - 3.0) * (extra / (extra + 100.0))
        # offset_c: 4 -> 0 as sensitivity grows
        offset_c = 4.0 - (4.0 - 0.0) * (extra / (extra + 100.0))

    block_size = int(round(block_size))
    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1

    offset_c = int(round(offset_c))
    offset_c = max(0, offset_c)

    p.adaptive_block_size = block_size
    p.adaptive_offset_c = offset_c
    return p


# =============================================================================
# Panels
# =============================================================================
def left_panel(session_defaults: Dict) -> Tuple[Optional[Image.Image], Optional[ROI], Optional[float]]:
    panel_open("1) Input & ROI")

    uploaded_img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="img_uploader")

    session_file = st.file_uploader("Load session JSON (optional)", type=["json"], key="session_uploader")
    if session_file is not None:
        loaded = load_session_json(session_file)
        if loaded:
            st.session_state["session_defaults"] = loaded
            st.success("Session loaded. (You can reuse saved ROI/settings.)")
        else:
            st.error("Could not read session JSON.")

    session_defaults = st.session_state.get("session_defaults", session_defaults or {})
    has_saved_roi = isinstance(session_defaults.get("roi"), dict)

    use_saved_roi = st.checkbox(
        "Use ROI from loaded session (skip drawing)",
        value=False,
        disabled=not has_saved_roi,
        help="Load a session JSON first to enable this option.",
    )

    if uploaded_img is None:
        st.info("Upload an image to start.")
        panel_close()
        return None, None, None

    img = Image.open(uploaded_img).convert("RGB")
    W, H = img.size
    disp, scale = resize_for_display(img, max_w=1100)

    # ROI shape: rectangle only
    draw_mode = "rect"

    cA, cB = st.columns([1, 1])
    with cA:
        if st.button("Clear drawing"):
            st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1
    with cB:
        st.caption("Draw ONE rectangle. Last shape wins.")

    roi: Optional[ROI] = None
    if use_saved_roi and has_saved_roi:
        try:
            # If a saved ROI exists but is not rect, ignore it and force redraw
            saved = session_defaults.get("roi", {})
            if str(saved.get("kind", "")).lower() != "rect":
                raise ValueError("Saved ROI is not a rect; redraw required.")
            roi = ROI(**saved).clamp(W, H)
            st.success("Using ROI from session.")
        except Exception:
            st.error("Saved ROI in session JSON is invalid or not rectangular. Please redraw.")
            roi = None
    else:
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=disp,
            update_streamlit=True,
            height=disp.size[1],
            width=disp.size[0],
            drawing_mode="rect",
            key=f"canvas_{st.session_state.get('canvas_key', 0)}",
        )
        if canvas.json_data:
            roi = roi_from_canvas(canvas.json_data, draw_mode="rect", scale=scale)
            if roi is not None:
                roi = roi.clamp(W, H)


    # Fine-adjust ROI
    if roi is not None:
        with st.expander("Fine-adjust ROI (keep shape)", expanded=False):
            if roi.kind == "rect":
                dx = st.slider("Expand X (px)", -500, 500, 0, key="roi_expand_x")
                dy = st.slider("Expand Y (px)", -500, 500, 0, key="roi_expand_y")
                roi.x0 -= dx
                roi.x1 += dx
                roi.y0 -= dy
                roi.y1 += dy
            else:
                dr = st.slider("Adjust radius (px)", -500, 500, 0, key="roi_adjust_r")
                assert roi.r is not None and roi.cx is not None and roi.cy is not None
                roi.r = max(1, roi.r + dr)
                roi.x0 = roi.cx - roi.r
                roi.x1 = roi.cx + roi.r
                roi.y0 = roi.cy - roi.r
                roi.y1 = roi.cy + roi.r
            roi = roi.clamp(W, H)

    panel_close()
    return img, roi, scale


def mid_panel(session_defaults: Dict) -> Params:
    panel_open("2) Parameters & Outputs")

    # Build params with session defaults (if any), then show UI controls.
    p = params_from_session(session_defaults)
    p = derive_threshold_params(p)

    p.min_area = st.number_input("Min deposit area (px)", min_value=1, max_value=50000, value=int(p.min_area))
    p.max_area = st.number_input("Max deposit area (px)", min_value=10, max_value=500000, value=int(p.max_area))

    # Sensitivity (unlimited): exact value input only (no slider)
    st.session_state.setdefault("sens_max", 200)

    p.sensitivity = int(
        st.number_input(
            "Sensitivity",
            min_value=0,
            value=int(getattr(p, "sensitivity", 0) or 0),
            step=1,
            help="Higher values generally detect more/stronger deposits (mapped to adaptive threshold params).",
            key="sensitivity_exact",
        )
    )

    # Keep sens_max in sync (optional, but safe if other parts rely on it)
    if p.sensitivity > int(st.session_state.sens_max):
        st.session_state.sens_max = int(max(int(st.session_state.sens_max) * 2, p.sensitivity))

    p.n_flies = st.number_input("No. flies", min_value=0, max_value=1000, value=int(p.n_flies), step=1)

    p.use_arenas = st.checkbox("Only count inside white arenas", value=bool(p.use_arenas))

    # Derived threshold mapping
    p = derive_threshold_params(p)

    with st.expander("Advanced", expanded=False):
        p.rod_circularity_cutoff = st.slider(
            "ROD circularity cutoff", 0.0, 1.0, float(p.rod_circularity_cutoff), 0.01
        )
        p.morph_open = st.slider("Morph OPEN", 0, 15, int(p.morph_open), 1)
        p.morph_close = st.slider("Morph CLOSE", 0, 25, int(p.morph_close), 1)

        if p.remove_pink_background:
            st.markdown("**Pink mask (HSV)**")
            p.pink_h_low = st.slider("Hue low", 0, 179, int(p.pink_h_low))
            p.pink_h_high = st.slider("Hue high", 0, 179, int(p.pink_h_high))
            p.pink_s_min = st.slider("Sat min", 0, 255, int(p.pink_s_min))
            p.pink_v_min = st.slider("Val min", 0, 255, int(p.pink_v_min))
            p.pink_close = st.slider("Close kernel", 0, 51, int(p.pink_close), 2)

        if p.use_arenas:
            st.markdown("**Arena mask (white circles)**")
            p.arena_s_max = st.slider("Arena S max", 0, 255, int(p.arena_s_max))
            p.arena_v_min = st.slider("Arena V min", 0, 255, int(p.arena_v_min))
            p.arena_close = st.slider("Arena close", 0, 35, int(p.arena_close))
            p.arena_erode = st.slider("Arena erode", 0, 51, int(p.arena_erode), 2)
            p.arena_min_area = st.slider("Arena min area", 1000, 500000, int(p.arena_min_area))
            p.arena_max_area = st.slider("Arena max area", 1000, 2000000, int(p.arena_max_area))

    with st.expander("Notes: how metrics are calculated", expanded=False):
        st.markdown(
            "- **Area / Perimeter / Circularity** are measured from each detected contour.  \n"
            "- **Circularity** = `4π × Area / Perimeter²` (1.0 is a perfect circle).  \n"
            "- **ROD** is flagged when Circularity < cutoff (more oblong deposits).  \n"
            "- **Mean RGB/HSL** values are *normalized to 0–1* and computed as the mean pixel intensity inside each deposit mask.  \n"
            "- **IOD (integrated optical density)** is computed as `Area × (1 − MeanL)` (darker deposits → larger IOD)."
        )
        st.caption(
            "Concise note: Using the background as a zero reference, color intensities are reported as normalized (0–1) "
            "mean RGB/HSL values within each detected deposit."
        )

    panel_close()
    return p


def compute_results(img: Image.Image, roi: ROI, p: Params):
    """Compute all outputs for current image+ROI+parameters."""
    W, H = img.size
    roi = roi.clamp(W, H)

    # Full ROI mask (for consistency), and crop
    roi_mask_full = make_roi_mask(W, H, roi)
    crop_img = img.crop((roi.x0, roi.y0, roi.x1, roi.y1))
    roi_mask_crop = roi_mask_full[roi.y0 : roi.y1, roi.x0 : roi.x1]

    # Display crops: for circle ROIs, hide everything outside the circle (alpha mask)
    if roi.kind == "circle":
        crop_display = apply_mask_alpha(crop_img, roi_mask_crop)
    else:
        crop_display = crop_img

    crop_bgr = pil_to_bgr(crop_img)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    bw = adaptive_threshold(gray, block_size=p.adaptive_block_size, c=p.adaptive_offset_c)
    bw = morphology(bw, open_k=int(p.morph_open), close_k=int(p.morph_close))
    bw = cv2.bitwise_and(bw, bw, mask=roi_mask_crop)

    pink_mask = None
    if p.remove_pink_background:
        pink_mask = pink_background_mask_hsv(
            crop_bgr,
            h_low=int(p.pink_h_low),
            h_high=int(p.pink_h_high),
            s_min=int(p.pink_s_min),
            v_min=int(p.pink_v_min),
            close_k=int(p.pink_close),
        )
        bw = cv2.bitwise_and(bw, bw, mask=cv2.bitwise_not(pink_mask))

    arena_mask = None
    if p.use_arenas:
        # Heuristic: if the ROI is already mostly "white", treat the ROI itself as the arena.
        roi_area = int(cv2.countNonZero(roi_mask_crop))
        white_candidate = white_candidate_mask_hsv(crop_bgr, s_max=int(p.arena_s_max), v_min=int(p.arena_v_min))
        white_in_roi = cv2.bitwise_and(white_candidate, white_candidate, mask=roi_mask_crop)
        white_area = int(cv2.countNonZero(white_in_roi))
        white_frac = (white_area / roi_area) if roi_area > 0 else 0.0

        if white_frac >= 0.75:
            # Common case: user draws ROI tightly around a single arena (all/mostly white).
            arena_mask = roi_mask_crop.copy()
        else:
            arena_mask = detect_arenas_white_regions(
                crop_bgr,
                s_max=int(p.arena_s_max),
                v_min=int(p.arena_v_min),
                close_k=int(p.arena_close),
                erode_k=int(p.arena_erode),
                min_area=int(p.arena_min_area),
                max_area=int(p.arena_max_area),
            )
            # Always restrict arena mask to the ROI to prevent accidental masking by outside regions.
            arena_mask = cv2.bitwise_and(arena_mask, arena_mask, mask=roi_mask_crop)

            # Fallback: if contour-based arena detection fails (empty), use the raw white mask inside ROI.
            if cv2.countNonZero(arena_mask) == 0:
                arena_mask = white_in_roi

        # Final safety net: never zero-out the whole ROI.
        if arena_mask is None or cv2.countNonZero(arena_mask) == 0:
            arena_mask = roi_mask_crop.copy()

        bw = cv2.bitwise_and(bw, bw, mask=arena_mask)

    df, _ = measure_spots_turd_style(
        crop_bgr,
        bw,
        min_area=int(p.min_area),
        max_area=int(p.max_area),
        rod_circ_cutoff=float(p.rod_circularity_cutoff),
    )
    summary = plate_summary(df, int(p.n_flies))
    ann = annotate(crop_bgr, bw, df)

    if roi.kind == "circle":
        ann_display = apply_mask_alpha(bgr_to_pil(ann), roi_mask_crop)
    else:
        ann_display = bgr_to_pil(ann)

    return {
        "roi": roi,
        "crop_img": crop_img,
        "crop_display": crop_display,
        "roi_mask_crop": roi_mask_crop,
        "bw": bw,
        "pink_mask": pink_mask,
        "arena_mask": arena_mask,
        "df": df,
        "summary": summary,
        "ann": ann,
        "ann_display": ann_display,
    }


def right_panel(img: Image.Image, roi: Optional[ROI], p: Params):
    panel_open("3) ROI & Overlay")

    if roi is None:
        st.info("Draw an ROI (left panel) to see results.")
        panel_close()
        return None

    results = compute_results(img, roi, p)

    tab1, tab2, tab3 = st.tabs(["Annotated", "Cropped ROI", "Masks"])
    with tab1:
        st.image(results["ann_display"], use_container_width=True)
    with tab2:
        st.image(results["crop_display"], use_container_width=True)
    with tab3:
        cX, cY = st.columns([1, 1])
        with cX:
            st.caption("Binary map")
            st.image(Image.fromarray(results["bw"]), use_container_width=True)
            st.caption("ROI mask")
            st.image(Image.fromarray(results["roi_mask_crop"]), use_container_width=True)
        with cY:
            if results["pink_mask"] is not None:
                st.caption("Pink mask")
                st.image(Image.fromarray(results["pink_mask"]), use_container_width=True)
            if results["arena_mask"] is not None:
                st.caption("Arena mask")
                st.image(Image.fromarray(results["arena_mask"]), use_container_width=True)

    st.markdown("**Quick summary**")
    st.dataframe(results["summary"], use_container_width=True)

    with st.expander("Deposits table", expanded=False):
        st.dataframe(results["df"], use_container_width=True)

    panel_close()
    return results


def bottom_bar(results, roi: Optional[ROI], p: Params) -> None:
    st.markdown('<div class="bar">', unsafe_allow_html=True)
    st.markdown("#### Save / Export")

    # Session JSON payload (even if results is None)
    session_payload = asdict(p)
    session_payload["roi"] = asdict(roi) if roi is not None else None

    # Buttons
    b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.2, 1.2])

    if results is None:
        # Keep the bar structure, but disable exports until ROI + processing exists.
        with b1:
            st.download_button("Download deposits CSV", b"", file_name="spots_turd_style.csv", disabled=True)
        with b2:
            st.download_button("Download summary CSV", b"", file_name="plate_summary_turd_style.csv", disabled=True)
        with b3:
            st.download_button("Download annotated PNG", b"", file_name="annotated_detection.png", disabled=True)
        with b4:
            st.download_button(
                "Download session JSON",
                json.dumps(session_payload, indent=2).encode("utf-8"),
                file_name="session_turd_style.json",
                mime="application/json",
            )
        st.markdown(
            '<div class="tiny">Tip: draw an ROI to enable CSV/PNG exports. You can still save a session JSON now.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = results["df"]
    summary = results["summary"]
    with b1:
        st.download_button(
            "Download deposits CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="spots_turd_style.csv",
            mime="text/csv",
        )
    with b2:
        st.download_button(
            "Download summary CSV",
            summary.to_csv(index=False).encode("utf-8"),
            file_name="plate_summary_turd_style.csv",
            mime="text/csv",
        )
    with b3:
        # Preserve alpha for circular ROIs (shows only inside the circle).
        buf = BytesIO()
        results["ann_display"].save(buf, format="PNG")
        ann_png = buf.getvalue()
        st.download_button(
            "Download annotated PNG",
            ann_png,
            file_name="annotated_detection.png",
            mime="image/png",
        )
    with b4:
        st.download_button(
            "Download session JSON",
            json.dumps(session_payload, indent=2).encode("utf-8"),
            file_name="session_turd_style.json",
            mime="application/json",
        )

    st.markdown(
        '<div class="tiny">Tip: Save “session JSON” → later reload it (left panel) → tick “Use ROI from loaded session”.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# App entry
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="TURD-like Spot Detector — Organized Layout", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.title("TURD-like Spot Detector")
    st.caption("Mouse ROI (rect/circle) → optional pink background removal → TURD-style measurements")

    # Session state defaults
    st.session_state.setdefault("canvas_key", 0)
    st.session_state.setdefault("session_defaults", {})

    # --- 3 panels ---
    left, mid, right = st.columns([1.25, 1.05, 1.1], gap="large")

    with left:
        img, roi, _scale = left_panel(st.session_state.get("session_defaults", {}))

    with mid:
        p = mid_panel(st.session_state.get("session_defaults", {}))

    results = None
    if img is not None:
        with right:
            results = right_panel(img, roi, p)

    bottom_bar(results, roi, p)


if __name__ == "__main__":
    main()
