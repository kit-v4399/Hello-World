# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
from io import BytesIO

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Image Processing Playground", layout="wide")
st.title("Image Processing Playground")
st.markdown(
    """
    ตัวอย่างแอปสำหรับทดลองการแปลงภาพ (grayscale / color) 
    โดยมีหลายเมธอด: linear negative, contrast stretching, piecewise linear,
    log, gamma, histogram equalization, adaptive histogram equalization (AHE), CLAHE.
    """
)

# ---------------------------
# Utility functions (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_image_file(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return img

@st.cache_data(show_spinner=False)
def pil_to_cv(img_pil):
    # PIL (RGB) -> OpenCV (BGR)
    arr = np.array(img_pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

@st.cache_data(show_spinner=False)
def cv_to_pil(img_cv):
    # OpenCV (BGR) or grayscale ndarray -> PIL (RGB or L)
    if img_cv.ndim == 2:
        return Image.fromarray(img_cv)
    bgr = img_cv
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def plot_histogram_image(img_arr, is_color=True, ax=None):
    """
    img_arr: numpy array in RGB if is_color True else 2D grayscale
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,2.5))
    else:
        fig = ax.figure
    ax.cla()
    if is_color and img_arr.ndim == 3:
        colors = ("r", "g", "b")
        for i, col in enumerate(colors):
            channel = img_arr[..., i].ravel()
            ax.hist(channel, bins=256, range=(0,255), color=col, alpha=0.6, density=False)
        ax.set_xlim(0,255)
        ax.set_title("Histogram (RGB channels)")
    else:
        ax.hist(img_arr.ravel(), bins=256, range=(0,255), color='k', alpha=0.7)
        ax.set_xlim(0,255)
        ax.set_title("Histogram (grayscale)")
    ax.grid(False)
    ax.set_ylabel("Count")
    ax.set_xlabel("Intensity")
    return fig

# ---------------------------
# Processing algorithms
# ---------------------------
@st.cache_data
def linear_negative_gray(img_gray):
    # img_gray: uint8 2D
    return 255 - img_gray

@st.cache_data
def linear_negative_color(img_bgr):
    # invert each channel
    return 255 - img_bgr

@st.cache_data
def contrast_stretching(img_arr, in_min=None, in_max=None):
    # works for grayscale 2D or color BGR 3D
    if img_arr.ndim == 2:
        a = in_min if in_min is not None else np.min(img_arr)
        b = in_max if in_max is not None else np.max(img_arr)
        out = exposure.rescale_intensity(img_arr, in_range=(a, b), out_range=(0,255)).astype(np.uint8)
        return out
    else:
        out = np.zeros_like(img_arr)
        for c in range(img_arr.shape[2]):
            a = in_min[c] if isinstance(in_min, (list, tuple, np.ndarray)) else (np.min(img_arr[..., c]) if in_min is None else in_min)
            b = in_max[c] if isinstance(in_max, (list, tuple, np.ndarray)) else (np.max(img_arr[..., c]) if in_max is None else in_max)
            out[..., c] = exposure.rescale_intensity(img_arr[..., c], in_range=(a, b), out_range=(0,255)).astype(np.uint8)
        return out

@st.cache_data
def piecewise_linear(img_arr, x_points, y_points):
    # x_points and y_points in 0..255 (lists), use np.interp
    lut = np.interp(np.arange(256), x_points, y_points).astype(np.uint8)
    return lut[img_arr]

@st.cache_data
def log_transform(img_arr, c=1.0):
    # img_arr uint8 -> normalize 0..1, compute log, scale back
    img_float = img_arr.astype(np.float32) / 255.0
    out = c * np.log1p(img_float)
    out = out / np.max(out)
    return (out * 255).astype(np.uint8)

@st.cache_data
def gamma_transform(img_arr, gamma=1.0, c=1.0):
    inv_gamma = 1.0 / gamma if gamma != 0 else 1.0
    img_float = img_arr.astype(np.float32) / 255.0
    out = c * (img_float ** inv_gamma)
    out = out / np.max(out) if np.max(out) != 0 else out
    return (out * 255).astype(np.uint8)

@st.cache_data
def histogram_equalization(img_arr):
    # For grayscale: cv2.equalizeHist
    # For color: transform Y channel in YCrCb
    if img_arr.ndim == 2:
        return cv2.equalizeHist(img_arr)
    else:
        ycrcb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

@st.cache_data
def adaptive_hist_equalization_skimage(img_arr, kernel_size=None, clip_limit=0.01):
    # Using skimage's adaptive histogram equalization (CLAHE alternative)
    # input expects float in [0,1], returns float [0,1]
    if img_arr.ndim == 2:
        out = exposure.equalize_adapthist(img_arr.astype(np.float32)/255.0, kernel_size=kernel_size, clip_limit=clip_limit)
        return (out * 255).astype(np.uint8)
    else:
        # Apply on each channel or use lab/ych model; apply per channel simpler
        out = np.zeros_like(img_arr)
        for c in range(3):
            channel = img_arr[..., c].astype(np.float32)/255.0
            res = exposure.equalize_adapthist(channel, kernel_size=kernel_size, clip_limit=clip_limit)
            out[..., c] = (res * 255).astype(np.uint8)
        return out

@st.cache_data
def clahe_cv2(img_arr, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    if img_arr.ndim == 2:
        return clahe.apply(img_arr)
    else:
        # Convert to LAB, apply on L channel
        lab = cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Input & Options")

upload_mode = st.sidebar.radio("Image type", ("Upload image", "Use sample image"))
color_mode = st.sidebar.radio("Process mode", ("Color", "Grayscale"))
method = st.sidebar.selectbox(
    "Choose processing method",
    [
        "Linear Negative",
        "Contrast Stretching",
        "Piecewise Linear",
        "Log Transformation",
        "Gamma Transformation",
        "Histogram Equalization",
        "Adaptive Histogram Equalization (AHE)",
        "CLAHE (OpenCV)"
    ],
)
# Upload image
uploaded_file = None
if upload_mode == "Upload image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
else:
    # sample image included from skimage or placeholder using OpenCV
    st.sidebar.info("Using built-in sample image.")
    # generate a sample gradient image or load from URL is disabled; create synthetic
    sample = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(sample, (0,0), (512,256), (100, 50, 200), -1)
    cv2.circle(sample, (256,256), 120, (200,200,50), -1)
    uploaded_file = BytesIO()
    Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)).save(uploaded_file, format="PNG")
    uploaded_file.seek(0)

# load image
if uploaded_file is None:
    st.warning("Please upload an image to proceed.")
    st.stop()

img_pil = load_image_file(uploaded_file)

# convert to desired mode
if color_mode == "Grayscale":
    img_pil_proc = ImageOps.grayscale(img_pil)
else:
    img_pil_proc = img_pil.copy()

# prepare numpy arrays for cv operations
if color_mode == "Grayscale":
    img_cv = np.array(img_pil_proc)  # 2D uint8
else:
    img_cv = pil_to_cv(img_pil_proc)  # BGR uint8

# ---------------------------
# Parameters for methods (dynamic)
# ---------------------------
st.sidebar.markdown("### Method parameters")
if method == "Linear Negative":
    st.sidebar.write("No parameters (invert intensity).")
elif method == "Contrast Stretching":
    st.sidebar.write("Select input range (percentile-based or auto).")
    use_percentile = st.sidebar.checkbox("Use percentiles", value=True)
    if use_percentile:
        p_low = st.sidebar.slider("Low percentile", 0, 50, 2)
        p_high = st.sidebar.slider("High percentile", 50, 100, 98)
    else:
        vmin = st.sidebar.slider("Input min (0-255)", 0, 255, 0)
        vmax = st.sidebar.slider("Input max (0-255)", 0, 255, 255)
elif method == "Piecewise Linear":
    st.sidebar.write("Define up to 4 control points (x: input, y: output).")
    # default linear points
    x1 = st.sidebar.number_input("x0", 0, 255, 0)
    y1 = st.sidebar.number_input("y0", 0, 255, 0)
    x2 = st.sidebar.number_input("x1", 0, 255, 64)
    y2 = st.sidebar.number_input("y1", 0, 255, 32)
    x3 = st.sidebar.number_input("x2", 0, 255, 192)
    y3 = st.sidebar.number_input("y2", 0, 255, 224)
    x4 = st.sidebar.number_input("x3", 0, 255, 255)
    y4 = st.sidebar.number_input("y3", 0, 255, 255)
elif method == "Log Transformation":
    c = st.sidebar.slider("Scaling constant c", 0.1, 10.0, 1.0)
elif method == "Gamma Transformation":
    gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0)
    c = st.sidebar.slider("Scaling constant c", 0.1, 2.0, 1.0)
elif method == "Histogram Equalization":
    st.sidebar.write("Global histogram equalization.")
elif method == "Adaptive Histogram Equalization (AHE)":
    clip = st.sidebar.slider("Clip limit", 0.001, 0.1, 0.01, format="%.3f")
    # kernel size: odd tuple => pass None or int
    use_kernel = st.sidebar.checkbox("Set kernel size", value=False)
    if use_kernel:
        ksize = st.sidebar.slider("kernel size (odd)", 3, 101, 8, step=2)
    else:
        ksize = None
elif method == "CLAHE (OpenCV)":
    clahe_clip = st.sidebar.slider("clipLimit", 1.0, 10.0, 2.0)
    tile_x = st.sidebar.slider("tileGridSize X", 2, 16, 8)
    tile_y = st.sidebar.slider("tileGridSize Y", 2, 16, 8)

# ---------------------------
# Run processing
# ---------------------------
st.markdown("---")
st.subheader("Before & After")

col1, col2 = st.columns([1,1])

with col1:
    st.markdown("**Original**")
    st.image(img_pil_proc, use_column_width=True)
    # histogram
    arr_display = np.array(img_pil_proc)
    fig_orig = plot_histogram_image(arr_display if color_mode == "Color" else arr_display, is_color=(color_mode=="Color"))
    st.pyplot(fig_orig)

# process
result_cv = None

# map methods to functions
if method == "Linear Negative":
    if color_mode == "Grayscale":
        result_cv = linear_negative_gray(img_cv)
    else:
        result_cv = linear_negative_color(img_cv)
elif method == "Contrast Stretching":
    if use_percentile:
        if color_mode == "Grayscale":
            low = np.percentile(img_cv, p_low)
            high = np.percentile(img_cv, p_high)
            result_cv = contrast_stretching(img_cv, in_min=low, in_max=high)
        else:
            low = [np.percentile(img_cv[..., c], p_low) for c in range(3)]
            high = [np.percentile(img_cv[..., c], p_high) for c in range(3)]
            result_cv = contrast_stretching(img_cv, in_min=low, in_max=high)
    else:
        if color_mode == "Grayscale":
            result_cv = contrast_stretching(img_cv, in_min=vmin, in_max=vmax)
        else:
            result_cv = contrast_stretching(img_cv, in_min=vmin, in_max=vmax)
elif method == "Piecewise Linear":
    # build mapping from control points
    x_points = [x1, x2, x3, x4]
    y_points = [y1, y2, y3, y4]
    # ensure increasing x - sort by x
    pts = sorted(zip(x_points, y_points), key=lambda p: p[0])
    xs, ys = zip(*pts)
    # clamp endpoints
    if xs[0] != 0:
        xs = (0,) + xs
        ys = (0,) + ys
    if xs[-1] != 255:
        xs = xs + (255,)
        ys = ys + (255,)
    if color_mode == "Grayscale":
        result_cv = piecewise_linear(img_cv, xs, ys)
    else:
        # apply per channel
        out = np.zeros_like(img_cv)
        for c in range(3):
            out[..., c] = piecewise_linear(img_cv[..., c], xs, ys)
        result_cv = out
elif method == "Log Transformation":
    result_cv = log_transform(img_cv, c=c)
elif method == "Gamma Transformation":
    result_cv = gamma_transform(img_cv, gamma=gamma, c=c)
elif method == "Histogram Equalization":
    result_cv = histogram_equalization(img_cv)
elif method == "Adaptive Histogram Equalization (AHE)":
    k = None if not use_kernel else ksize
    result_cv = adaptive_hist_equalization_skimage(img_cv, kernel_size=k, clip_limit=clip)
elif method == "CLAHE (OpenCV)":
    result_cv = clahe_cv2(img_cv, clipLimit=clahe_clip, tileGridSize=(tile_x, tile_y))
else:
    st.error("Method not implemented.")
    st.stop()

# convert result for display
if color_mode == "Grayscale":
    result_pil = cv_to_pil(result_cv)
else:
    result_pil = cv_to_pil(result_cv)  # cv_to_pil handles BGR -> RGB

with col2:
    st.markdown(f"**Result — {method}**")
    st.image(result_pil, use_column_width=True)
    # histogram
    arr_result = np.array(result_pil)
    fig_res = plot_histogram_image(arr_result if color_mode == "Color" else arr_result, is_color=(color_mode=="Color"))
    st.pyplot(fig_res)

# ---------------------------
# Side outputs: download result or show summary
# ---------------------------
st.markdown("---")
c1, c2 = st.columns([1,3])

with c1:
    buffer = BytesIO()
    # save as PNG
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button("Download Result (PNG)", data=buffer, file_name="result.png", mime="image/png")

with c2:
    st.markdown("**Processing summary**")
    st.write(f"- Method: **{method}**")
    st.write(f"- Mode: **{color_mode}**")
    # parameter summary
    if method == "Contrast Stretching":
        if use_percentile:
            st.write(f"- Using percentiles: low={p_low}%, high={p_high}%")
        else:
            st.write(f"- Input range: min={vmin}, max={vmax}")
    elif method == "Piecewise Linear":
        st.write(f"- Control points: {list(zip(xs, ys))}")
    elif method == "Log Transformation":
        st.write(f"- c = {c}")
    elif method == "Gamma Transformation":
        st.write(f"- gamma = {gamma}, c = {c}")
    elif method == "Adaptive Histogram Equalization (AHE)":
        st.write(f"- clip_limit = {clip}, kernel = {ksize}")
    elif method == "CLAHE (OpenCV)":
        st.write(f"- clipLimit = {clahe_clip}, tileGridSize = ({tile_x}, {tile_y})")

st.markdown("### Notes")
st.markdown(
    """
    - สำหรับภาพสี (Color) บางวิธี (เช่น histogram equalization, CLAHE) เราทำงานบนช่องความสว่าง (Y/L) แทนที่จะปรับทุกช่องแยกกัน เพื่อรักษาสีของภาพ  
    - การใช้พารามิเตอร์ที่สูง/ต่ำเกินไปอาจทำให้เกิด artifacts หรือสูญเสียรายละเอียดได้ ให้ลองปรับค่าอย่างระมัดระวัง  
    - ฟังก์ชันบางตัวถูก cached เพื่อเพิ่มประสิทธิภาพเมื่อทดลองซ้ำ ๆ
    """
)

