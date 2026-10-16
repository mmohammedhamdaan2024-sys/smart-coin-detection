"""
Smart Coin Detection, Counting, Classification, and Condition Analysis System
Digital Image Processing Web Application built with Streamlit and OpenCV.
"""

import os
import sys
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# Ensure local module path resolution
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import (
    DEFAULT_MIN_DIAMETER_PX,
    DEFAULT_MAX_DIAMETER_PX,
    SAMPLE_IMAGES_DIR,
    COLOR_PALETTE
)
from modules.image_loader import ImageLoader
from modules.coin_detector import CoinDetector
from modules.reference_database import ReferenceDatabase
from modules.report_generator import ReportGenerator
from modules.history_manager import HistoryManager


# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart Coin Detection System",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Computer Vision Dashboard Theme
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid #0D9488;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-subtitle {
        color: #CBD5E1;
        font-size: 0.80rem;
        margin-top: 2px;
    }
    /* Section Headers */
    .section-header {
        color: #0284C7;
        font-size: 1.35rem;
        font-weight: 700;
        border-bottom: 2px solid #0284C7;
        padding-bottom: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    /* Info Callout */
    .academic-disclaimer {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #E2E8F0;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "current_image_meta" not in st.session_state:
    st.session_state.current_image_meta = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "selected_coin_id" not in st.session_state:
    st.session_state.selected_coin_id = 1
if "manual_coin_data" not in st.session_state:
    st.session_state.manual_coin_data = {}
if "last_analyzed_hash" not in st.session_state:
    st.session_state.last_analyzed_hash = ""
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "2. Analyze New Image" if st.session_state.analysis_result is None else "1. Dashboard"
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = ""
if "authenticated_role" not in st.session_state:
    st.session_state.authenticated_role = ""


# ==========================================
# REUSABLE MANAGERS
# ==========================================
@st.cache_resource
def get_ref_db():
    return ReferenceDatabase()

@st.cache_resource
def get_history_mgr():
    return HistoryManager()

ref_db = get_ref_db()
history_mgr = get_history_mgr()
image_loader = ImageLoader()


# ==========================================
# AUTHENTICATION & LOGIN GATEWAY
# ==========================================
if not st.session_state.is_authenticated:
    st.markdown("<br>", unsafe_allow_html=True)
    c_pad1, c_box, c_pad2 = st.columns([1, 2.2, 1])

    with c_box:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="font-size: 3.8rem;">🪙</span>
            <h1 style="color: #0284C7; margin-bottom: 2px;">Smart Coin Detection System</h1>
            <p style="color: #94A3B8; font-size: 1.05rem;">Digital Image Processing & Computer Vision Laboratory</p>
        </div>
        """, unsafe_allow_html=True)

        login_tab1, login_tab2 = st.tabs(["🔐 Sign In with Email", "⚡ Quick Sample Logins"])

        with login_tab1:
            st.markdown("##### Sign in to access the system")
            user_email = st.text_input(
                "Email ID",
                placeholder="e.g. faculty@university.edu or your.email@gmail.com",
                help="Enter your institutional or personal email address."
            )
            user_password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password (minimum 4 characters)",
                help="Any password with at least 4 characters is accepted for demonstration access."
            )

            col_auth_btn, col_help = st.columns([1.2, 1])
            with col_auth_btn:
                submit_login = st.button("🚀 Sign In to Dashboard", type="primary", use_container_width=True)

            if submit_login:
                email_clean = user_email.strip()
                if not email_clean or "@" not in email_clean or "." not in email_clean:
                    st.error("Please enter a valid Email ID (e.g. faculty@university.edu or yourname@gmail.com).")
                elif len(user_password) < 4:
                    st.error("Password must be at least 4 characters long.")
                else:
                    st.session_state.is_authenticated = True
                    st.session_state.authenticated_user = email_clean
                    role_tag = "Faculty / Evaluator" if ("faculty" in email_clean.lower() or "prof" in email_clean.lower() or ".edu" in email_clean.lower()) else "Reviewer"
                    st.session_state.authenticated_role = role_tag
                    st.success(f"Access granted! Welcome, {email_clean}")
                    st.rerun()

            st.caption("ℹ️ Anyone (faculty, neighbors, friends) can log in with their email address.")

        with login_tab2:
            st.markdown("##### 1-Click Demo Profiles for WhatsApp Reviewers")
            st.caption("Tap any profile below to immediately fill and enter the application:")

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                if st.button("🎓 Faculty\nfaculty@university.edu", use_container_width=True):
                    st.session_state.is_authenticated = True
                    st.session_state.authenticated_user = "faculty@university.edu"
                    st.session_state.authenticated_role = "Faculty / Professor"
                    st.rerun()
            with col_f2:
                if st.button("💻 Student\nstudent@university.edu", use_container_width=True):
                    st.session_state.is_authenticated = True
                    st.session_state.authenticated_user = "student@university.edu"
                    st.session_state.authenticated_role = "Student Developer"
                    st.rerun()
            with col_f3:
                if st.button("👤 Neighbor / Guest\nguest@reviewer.com", use_container_width=True):
                    st.session_state.is_authenticated = True
                    st.session_state.authenticated_user = "guest@reviewer.com"
                    st.session_state.authenticated_role = "Guest Reviewer"
                    st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.82rem;">
            Course: B.Sc. Computer Science | Digital Image Processing Final Project<br>
            Powered by OpenCV, NumPy, and Streamlit
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ==========================================
# SIDEBAR NAVIGATION & SETTINGS
# ==========================================
NAV_OPTIONS = [
    "1. Dashboard",
    "2. Analyze New Image",
    "3. Coin Detection Results",
    "4. Coin Condition Analysis",
    "5. Image Processing Visualization",
    "6. Analysis History",
    "7. Export Reports",
    "8. About the Project"
]

with st.sidebar:
    st.markdown("## 🪙 Coin Vision AI")
    st.caption("Digital Image Processing Lab")

    # User Profile & Logout
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        st.caption(f"👤 **{st.session_state.authenticated_user}**\n*{st.session_state.authenticated_role}*")
    with col_u2:
        if st.button("Logout", key="btn_logout"):
            st.session_state.is_authenticated = False
            st.session_state.authenticated_user = ""
            st.session_state.authenticated_role = ""
            st.rerun()

    st.markdown("---")

    current_idx = NAV_OPTIONS.index(st.session_state.nav_page) if st.session_state.nav_page in NAV_OPTIONS else 1
    nav_choice = st.radio(
        "Navigation",
        NAV_OPTIONS,
        index=current_idx
    )
    st.session_state.nav_page = nav_choice

    st.markdown("---")
    st.subheader("⚙️ Detection Settings")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        min_diam = st.number_input("Min Diam (px)", min_value=15, max_value=300, value=DEFAULT_MIN_DIAMETER_PX, step=5)
    with col_s2:
        max_diam = st.number_input("Max Diam (px)", min_value=50, max_value=800, value=DEFAULT_MAX_DIAMETER_PX, step=10)

    sensitivity = st.slider("Hough Sensitivity", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="Higher detects fainter coins; lower avoids noise.")
    enable_watershed = st.checkbox("Watershed Segmentation", value=True, help="Separates touching or overlapping coins.")
    enable_clahe = st.checkbox("CLAHE Contrast Enhancement", value=True, help="Equalizes local contrast for low-light images.")

    st.markdown("---")
    st.subheader("📐 Physical Calibration")
    calib_mode = st.radio("Calibration Scale", ["None (Pixels only)", "Custom mm/px Ratio", "Reference Coin Size"], index=0)
    
    calibration_scale = None
    if calib_mode == "Custom mm/px Ratio":
        custom_ratio = st.number_input("Millimeters per pixel (mm/px)", min_value=0.01, max_value=2.0, value=0.15, step=0.01, format="%.4f")
        calibration_scale = custom_ratio
    elif calib_mode == "Reference Coin Size":
        known_mm = st.number_input("Known Reference Diameter (mm)", min_value=10.0, max_value=50.0, value=24.26, step=0.1, help="e.g. US Quarter = 24.26mm")
        measured_ref_px = st.number_input("Measured Reference Diameter (px)", min_value=15.0, max_value=600.0, value=160.0, step=1.0)
        if measured_ref_px > 0:
            calibration_scale = known_mm / measured_ref_px
            st.caption(f"Scale: **{calibration_scale:.4f} mm/px**")

    st.markdown("---")
    st.subheader("🏛️ Currency Filter")
    currencies = ["ALL"] + ref_db.get_currencies()
    preferred_currency = st.selectbox("Preferred Currency", currencies, index=0)


# ==========================================
# VIEW 1: MAIN DASHBOARD
# ==========================================
if nav_choice == "1. Dashboard":
    st.title("🪙 Smart Coin Detection System")
    st.markdown("##### *AI-Assisted Coin Recognition, Counting and Condition Analysis Using Digital Image Processing*")

    if st.session_state.analysis_result is None:
        st.info("👋 Welcome! No image has been analyzed yet. Go to **Analyze New Image** in the sidebar to upload an image or select a preset sample.")
    else:
        res = st.session_state.analysis_result
        stats = res["summary_stats"]
        q_metrics = res["quality_metrics"]

        # Main KPI Cards
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Detected</div>
                <div class="metric-value">{stats['total_coins']}</div>
                <div class="metric-subtitle">Verified coin contours</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Classified</div>
                <div class="metric-value">{stats['classified_coins']} / {stats['total_coins']}</div>
                <div class="metric-subtitle">Denominations matched</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            total_val = stats['total_monetary_value']
            curr_code = stats['monetary_currency']
            val_text = f"{curr_code} {total_val:.2f}" if (total_val > 0 and curr_code != "N/A") else "Unassessed"
            if stats['is_monetary_partial'] and total_val > 0:
                val_text += " *"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Face Value</div>
                <div class="metric-value">{val_text}</div>
                <div class="metric-subtitle">{'* Partial confirmed total' if stats['is_monetary_partial'] else 'Confident identification'}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Visible Anomalies</div>
                <div class="metric-value">{stats['anomaly_coins_count']}</div>
                <div class="metric-subtitle">Scratches / oxidation</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Diameter</div>
                <div class="metric-value">{stats['avg_diameter_px']:.1f} <span style="font-size:1rem;">px</span></div>
                <div class="metric-subtitle">Circularity: {stats['avg_circularity']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Image Quality</div>
                <div class="metric-value">{q_metrics['quality_rating'].split()[0]}</div>
                <div class="metric-subtitle">Blur: {q_metrics['blur_score']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Overview Visuals
        col_main_img, col_main_breakdown = st.columns([3, 2])

        with col_main_img:
            st.markdown('<div class="section-header">Annotated Detection Overview</div>', unsafe_allow_html=True)
            st.image(res["annotated_rgb"], use_container_width=True)
            if res["has_touching_coins"]:
                st.warning("⚠️ Touching or overlapping coins detected. Watershed boundary separation was applied.")

        with col_main_breakdown:
            st.markdown('<div class="section-header">Denomination Breakdown</div>', unsafe_allow_html=True)
            if stats["denomination_counts"]:
                df_denoms = pd.DataFrame(
                    list(stats["denomination_counts"].items()),
                    columns=["Denomination", "Count"]
                )
                st.dataframe(df_denoms, hide_index=True, use_container_width=True)
            else:
                st.info("No coins detected.")

            st.markdown('<div class="section-header">Surface Condition Distribution</div>', unsafe_allow_html=True)
            if stats["condition_counts"]:
                df_conds = pd.DataFrame(
                    list(stats["condition_counts"].items()),
                    columns=["Condition Screening Category", "Count"]
                )
                st.dataframe(df_conds, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="academic-disclaimer">
            <strong>Academic Note:</strong> Total face value is aggregated strictly from coins whose denomination is positively matched.
            Coins marked as <em>'Denomination uncertain'</em> are never assigned an arbitrary value.
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# VIEW 2: ANALYZE NEW IMAGE (Upload & Run)
# ==========================================
elif nav_choice == "2. Analyze New Image":
    st.title("📤 Analyze New Image")
    st.markdown("Upload a photo containing coins or choose from preset laboratory test scenarios.")

    col_up, col_preset = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload coin image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
            help="Supports high-resolution single or multi-coin photos."
        )

    with col_preset:
        st.markdown("**Or load a preset laboratory test image:**")
        preset_options = {
            "Select sample...": None,
            "1. Single Coin (Silver Quarter)": "sample_1_coin.jpg",
            "2. Three Separated Coins (Cent, Nickel, Quarter)": "sample_3_coins.jpg",
            "3. Five Mixed Coins (All denominations)": "sample_5_coins.jpg",
            "4. Touching Coins (Edge contact)": "sample_touching_coins.jpg",
            "5. Overlapping Coins (Partial occlusion)": "sample_overlapping_coins.jpg",
            "6. Blank Scene (No coins present)": "sample_no_coins.jpg",
            "7. Blurry Coin (Optical blur test)": "sample_blurry_coin.jpg",
            "8. Low-Light Image (Severe underexposure)": "sample_low_light.jpg",
            "9. Damaged Coin (Scratches & oxidation)": "sample_damaged_coin.jpg"
        }
        chosen_preset = st.selectbox("Preset Test Scenarios", list(preset_options.keys()))

    # Determine image source
    active_source = None
    file_label = ""
    if uploaded_file is not None:
        active_source = uploaded_file
        file_label = uploaded_file.name
    elif chosen_preset != "Select sample..." and preset_options[chosen_preset] is not None:
        preset_path = SAMPLE_IMAGES_DIR / preset_options[chosen_preset]
        if preset_path.exists():
            active_source = str(preset_path)
            file_label = preset_options[chosen_preset]

    if active_source is not None:
        loaded = image_loader.load_image(active_source, file_name=file_label)

        if not loaded["is_valid"]:
            st.error(f"❌ Error loading image: {loaded['error_message']}")
        else:
            st.session_state.current_image_meta = loaded

            # Image Info & Preview Row
            col_preview, col_meta = st.columns([2, 1])

            with col_preview:
                st.image(loaded["rgb"], caption=f"Loaded Input: {loaded['file_name']}", use_container_width=True)

            with col_meta:
                st.markdown("### 📋 Image Metadata")
                st.write(f"**Filename:** `{loaded['file_name']}`")
                st.write(f"**Original Size:** {loaded['original_width']} × {loaded['original_height']} px")
                st.write(f"**Processed Resolution:** {loaded['width']} × {loaded['height']} px")
                st.write(f"**File Size:** {loaded['file_size_kb']} KB")
                st.write(f"**Scale Factor:** {loaded['scale_factor']:.2f}")

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    analyze_btn = st.button("🚀 Analyze Coins", type="primary", use_container_width=True)
                with btn_col2:
                    if st.button("🔄 Reset Image", use_container_width=True):
                        st.session_state.current_image_meta = None
                        st.session_state.analysis_result = None
                        st.rerun()

            if analyze_btn:
                with st.spinner("Processing digital image pipeline (Denoising, CLAHE, Hough, Watershed, Geometry)..."):
                    detector = CoinDetector(
                        reference_db=ref_db,
                        calibration_scale=calibration_scale
                    )
                    analysis = detector.detect_and_analyze(
                        bgr_image=loaded["bgr"],
                        min_diameter_px=min_diam,
                        max_diameter_px=max_diam,
                        detection_sensitivity=sensitivity,
                        enable_watershed_split=enable_watershed,
                        preferred_currency=preferred_currency,
                        manual_coin_data=st.session_state.manual_coin_data
                    )
                    st.session_state.analysis_result = analysis
                    st.session_state.selected_coin_id = 1 if analysis["total_detected"] > 0 else None

                    # Automatically record in SQLite history
                    history_mgr.save_analysis(
                        filename=loaded["file_name"],
                        width=loaded["width"],
                        height=loaded["height"],
                        summary_stats=analysis["summary_stats"],
                        config_parameters={
                            "min_diam_px": min_diam,
                            "max_diam_px": max_diam,
                            "sensitivity": sensitivity,
                            "watershed": enable_watershed,
                            "clahe": enable_clahe,
                            "calibration_scale": calibration_scale
                        },
                        coins_list=analysis["coins"]
                    )

                st.success(f"Analysis completed successfully! Detected {analysis['total_detected']} coin(s).")
                st.rerun()

            # Display marked results on the current page if analysis is available
            if st.session_state.analysis_result is not None:
                st.markdown("---")
                res = st.session_state.analysis_result
                st.markdown(f'<div class="section-header">Detection Results: {res["total_detected"]} Coin(s) Identified</div>', unsafe_allow_html=True)

                col_det_img, col_det_tbl = st.columns([3, 2])
                with col_det_img:
                    st.image(res["annotated_rgb"], caption=f"Annotated Detections ({loaded['file_name']})", use_container_width=True)

                with col_det_tbl:
                    st.markdown("#### 📊 Measured Coins & Diameters")
                    table_rows = []
                    for c in res["coins"]:
                        geo = c["geometry"]
                        cl = c["classification"]
                        table_rows.append({
                            "ID": f"#{c['id']}",
                            "Diameter (px)": f"{geo['diameter_px']:.1f} px",
                            "Diameter (mm)": f"{geo['diameter_mm']} mm" if geo['diameter_mm'] else "Uncalibrated",
                            "Circularity": f"{geo['circularity']:.3f}",
                            "Denomination": cl.get("denomination", "Uncertain")
                        })
                    if table_rows:
                        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
                    else:
                        st.info("No coins detected in this image.")

                    c_nav1, c_nav2 = st.columns(2)
                    with c_nav1:
                        if st.button("🔍 Inspect Individual Coins", type="primary", use_container_width=True):
                            st.session_state.nav_page = "3. Coin Detection Results"
                            st.rerun()
                    with c_nav2:
                        if st.button("📊 Open Dashboard", use_container_width=True):
                            st.session_state.nav_page = "1. Dashboard"
                            st.rerun()


# ==========================================
# VIEW 3: COIN DETECTION RESULTS (Interactive)
# ==========================================
elif nav_choice == "3. Coin Detection Results":
    st.title("🎯 Coin Detection Results")

    if st.session_state.analysis_result is None:
        st.warning("No analysis available. Please upload and analyze an image first.")
    else:
        res = st.session_state.analysis_result
        coins = res["coins"]

        if not coins:
            st.info("No coins were identified in this image. Try adjusting detection sensitivity or diameter thresholds.")
        else:
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown('<div class="section-header">Interactive Coin Inspector</div>', unsafe_allow_html=True)
                
                # Coin selector
                coin_ids = [c["id"] for c in coins]
                selected_id = st.selectbox(
                    "Select Coin to Inspect & Highlight:",
                    coin_ids,
                    index=coin_ids.index(st.session_state.selected_coin_id) if st.session_state.selected_coin_id in coin_ids else 0
                )
                st.session_state.selected_coin_id = selected_id

                # Generate highlighted annotated image
                detector = CoinDetector(ref_db, res.get("calibration_scale"))
                highlighted_bgr = detector.generate_annotated_image(
                    bgr_image=st.session_state.current_image_meta["bgr"],
                    detected_coins=coins,
                    selected_coin_id=selected_id
                )
                st.image(cv2.cvtColor(highlighted_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col_right:
                # Retrieve currently selected coin
                selected_coin = next(c for c in coins if c["id"] == selected_id)
                geo = selected_coin["geometry"]
                cl = selected_coin["classification"]
                wt = selected_coin["weight"]
                age = selected_coin["age"]

                st.markdown(f'<div class="section-header">Coin #{selected_id} Specifications</div>', unsafe_allow_html=True)

                # Show Crop Image and Download Button
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    st.image(selected_coin["crop_rgb"], caption=f"Coin #{selected_id} Optical Crop", width=180)
                with col_c2:
                    # Provide download for individual crop
                    crop_pil = Image.fromarray(selected_coin["crop_rgb"])
                    crop_buf = io.BytesIO()
                    crop_pil.save(crop_buf, format="PNG")
                    st.download_button(
                        f"⬇️ Download Crop #{selected_id}",
                        data=crop_buf.getvalue(),
                        file_name=f"coin_{selected_id}_crop.png",
                        mime="image/png"
                    )
                    st.markdown(f"**Detection Confidence:** {selected_coin['detection_confidence']}%")
                    st.markdown(f"**Status:** `{selected_coin['detection_status']}`")

                # Geometry Tabs
                st.markdown("#### 📐 Geometric Measurements")
                g_df = pd.DataFrame([
                    {"Parameter": "Center Coordinates (x, y)", "Value": f"{geo['center_px']}"},
                    {"Parameter": "Radius (px)", "Value": f"{geo['radius_px']} px"},
                    {"Parameter": "Diameter (px)", "Value": f"{geo['diameter_px']} px"},
                    {"Parameter": "Physical Diameter (mm)", "Value": f"{geo['diameter_mm']} mm" if geo['diameter_mm'] else "Uncalibrated"},
                    {"Parameter": "Contour Area", "Value": f"{geo['contour_area_px2']} px²"},
                    {"Parameter": "Perimeter", "Value": f"{geo['perimeter_px']} px"},
                    {"Parameter": "Circularity Index", "Value": f"{geo['circularity']:.4f} (Ideal = 1.0)"},
                    {"Parameter": "Aspect Ratio", "Value": f"{geo['aspect_ratio']:.3f}"},
                    {"Parameter": "Orientation Angle", "Value": f"{geo['orientation_deg']}°"}
                ])
                st.dataframe(g_df, hide_index=True, use_container_width=True)

                # Classification & Denomination Card
                st.markdown("#### 🏛️ Denomination & Currency")
                if cl["is_classified"]:
                    st.success(f"**Identified:** {cl['denomination']} ({cl['currency']})")
                    st.write(f"**Issuing Country:** {cl['country']}")
                    st.write(f"**Face Value:** {cl['currency_symbol']}{cl['face_value']:.2f}")
                    st.write(f"**Classification Confidence:** {cl['classification_confidence']}%")
                else:
                    st.warning(f"**{cl['denomination']}**")
                    st.caption(cl["notes"])

                if cl["top_candidates"]:
                    with st.expander("View Candidate Denominations"):
                        cand_df = pd.DataFrame(cl["top_candidates"])
                        st.dataframe(cand_df, hide_index=True)

                # Weight & Mass Section
                st.markdown("#### ⚖️ Weight & Mass Analysis")
                st.info(f"**Status:** {wt['weight_status']}\n\n{wt['display_text']}")
                st.caption(f"ℹ️ {wt['method_notes']}")

                # User manual scale entry option
                with st.expander("Record Scale Measurement (Grams)"):
                    current_m = st.session_state.manual_coin_data.get(selected_id, {}).get("manual_weight_g", 0.0)
                    user_w = st.number_input(f"Enter Scale Mass for Coin #{selected_id} (g):", min_value=0.0, max_value=50.0, value=float(current_m), step=0.01)
                    if st.button(f"Save Weight for #{selected_id}"):
                        if selected_id not in st.session_state.manual_coin_data:
                            st.session_state.manual_coin_data[selected_id] = {}
                        st.session_state.manual_coin_data[selected_id]["manual_weight_g"] = user_w
                        st.success(f"Saved {user_w}g for Coin #{selected_id}. Click 'Analyze Coins' to update calculations.")

                # History & Age Section
                st.markdown("#### ⏳ Minting Year & Chronological Age")
                st.write(f"**Age Status:** {age['age_status']}")
                st.write(f"**Historical Notes:** {age['display_text']}")
                st.caption(age["notes"])

            # Full Detected Coins Catalog Table
            st.markdown("---")
            st.markdown('<div class="section-header">Complete Detected Coins Inventory</div>', unsafe_allow_html=True)
            all_coins_rows = []
            for c in coins:
                g = c["geometry"]
                cl_item = c["classification"]
                cd = c["condition"]
                wt_item = c["weight"]
                all_coins_rows.append({
                    "Coin ID": f"#{c['id']}",
                    "Status": c["detection_status"],
                    "Diameter (px)": f"{g['diameter_px']:.1f} px",
                    "Diameter (mm)": f"{g['diameter_mm']} mm" if g['diameter_mm'] else "Uncalibrated",
                    "Circularity": f"{g['circularity']:.4f}",
                    "Area (px²)": f"{g['contour_area_px2']:.0f}",
                    "Denomination": cl_item.get("denomination", "Uncertain"),
                    "Confidence": f"{cl_item.get('classification_confidence', 0)}%",
                    "Condition": cd.get("condition_category", "N/A"),
                    "Ref Mass": f"{wt_item.get('reference_mass_g')} g" if wt_item.get('reference_mass_g') else "N/A"
                })
            st.dataframe(pd.DataFrame(all_coins_rows), hide_index=True, use_container_width=True)


# ==========================================
# VIEW 4: COIN CONDITION ANALYSIS
# ==========================================
elif nav_choice == "4. Coin Condition Analysis":
    st.title("🔬 Surface Condition & Visible Anomaly Screening")
    st.markdown("Image-based defect screening detects surface scratches, corrosion flecks, tarnish, and edge irregularities.")

    st.markdown("""
    <div class="academic-disclaimer">
        <strong>Numismatic Grading Disclaimer:</strong> Image-based condition analysis is an automated visual screening tool, 
        not a professional coin-grading or authentication service. Professional grading requires microscopic physical inspection, 
        luster tilt analysis, and metal authenticity testing.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.analysis_result is None:
        st.warning("Please analyze an image first.")
    else:
        coins = st.session_state.analysis_result["coins"]
        q = st.session_state.analysis_result["quality_metrics"]

        st.markdown(f"**Image Quality Assessment:** `{q['quality_rating']}` | Sharpness Variance: `{q['blur_score']:.1f}` | Saturated Pixels: `{q['overexposed_pct']}%`")

        for c in coins:
            cid = c["id"]
            cond = c["condition"]

            with st.container():
                st.markdown(f"### Coin #{cid} — {cond['condition_category']}")
                
                col_c1, col_c2, col_c3 = st.columns([1, 1, 2])

                with col_c1:
                    st.image(c["crop_rgb"], caption=f"Coin #{cid} Optical View", width=180)

                with col_c2:
                    if cond["annotated_crop"] is not None:
                        st.image(cond["annotated_crop"], caption="Anomaly Mask (Red Overlay)", width=180)

                with col_c3:
                    st.markdown(f"**Candidate Anomaly Regions:** `{cond['anomaly_count']}`")
                    st.markdown(f"**Affected Visible Field:** `{cond['anomaly_area_pct']}%`")
                    st.markdown(f"**Edge / Rim Irregularity:** `{'Detected' if cond.get('has_edge_irregularity') else 'None'}`")
                    st.markdown(f"**Screening Confidence:** `{cond['confidence_score']}%`")
                    st.info(f"**Summary:** {cond['screening_summary']}")

                st.markdown("---")


# ==========================================
# VIEW 5: IMAGE PROCESSING PIPELINE VISUALIZATION
# ==========================================
elif nav_choice == "5. Image Processing Visualization":
    st.title("🔬 Image Processing Pipeline Visualization")
    st.markdown("Step-by-step digital image processing transformations implemented with OpenCV and NumPy.")

    if st.session_state.analysis_result is None:
        st.warning("Please run an analysis to view the pipeline transformations.")
    else:
        stages = st.session_state.analysis_result["pipeline_stages"]

        stage_keys = [
            "1_original",
            "2_grayscale",
            "3_denoised",
            "4_contrast_enhanced",
            "5_thresholded",
            "6_edges",
            "7_morphology",
            "8_segmentation",
            "9_final_detection"
        ]

        tabs = st.tabs([f"Stage {i+1}: {k.split('_', 1)[1].replace('_', ' ').title()}" for i, k in enumerate(stage_keys)])

        for idx, key in enumerate(stage_keys):
            with tabs[idx]:
                if key in stages:
                    stage = stages[key]
                    st.markdown(f"### {stage['title']}")
                    
                    col_img, col_desc = st.columns([3, 2])

                    with col_img:
                        if stage["is_gray"]:
                            st.image(stage["image"], caption=stage["title"], clamp=True, use_container_width=True)
                        else:
                            st.image(stage["image"], caption=stage["title"], use_container_width=True)

                    with col_desc:
                        st.markdown(f"**Algorithm:** `{stage['algorithm']}`")
                        st.markdown("#### 🎯 Purpose")
                        st.write(stage["purpose"])
                        st.markdown("#### ⚙️ Technical Explanation")
                        st.write(stage["explanation"])
                        st.markdown("#### 📊 Operational Parameters")
                        st.code(stage["parameters"], language="text")

        # Side-by-Side Comparison
        st.markdown('<div class="section-header">Direct Comparison: Original vs Final Detection</div>', unsafe_allow_html=True)
        col_orig, col_final = st.columns(2)
        with col_orig:
            st.image(stages["1_original"]["image"], caption="Original Photographic Ingestion", use_container_width=True)
        with col_final:
            st.image(stages["9_final_detection"]["image"], caption="Annotated Detection Deliverable", use_container_width=True)


# ==========================================
# VIEW 6: ANALYSIS HISTORY (SQLite)
# ==========================================
elif nav_choice == "6. Analysis History":
    st.title("📜 Local Analysis History")
    st.markdown("Historical audit trail persisted in local SQLite database (`outputs/history.db`).")

    history_df = history_mgr.get_all_records()

    if history_df.empty:
        st.info("No saved analysis runs in SQLite history yet.")
    else:
        st.dataframe(history_df, use_container_width=True)

        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            rec_id_to_inspect = st.selectbox("Inspect or Reopen History Record ID:", history_df["id"].tolist())
            if st.button("🔍 View Record Details"):
                record = history_mgr.get_record_by_id(rec_id_to_inspect)
                if record:
                    st.json(record)

        with col_h2:
            st.markdown("#### Database Maintenance")
            if st.button("🗑️ Delete Selected Record"):
                if history_mgr.delete_record(rec_id_to_inspect):
                    st.success(f"Deleted record #{rec_id_to_inspect}")
                    st.rerun()
            if st.button("⚠️ Clear Entire History Table"):
                history_mgr.clear_history()
                st.success("History cleared.")
                st.rerun()


# ==========================================
# VIEW 7: EXPORT REPORTS
# ==========================================
elif nav_choice == "7. Export Reports":
    st.title("📥 Export Analysis Reports")
    st.markdown("Download full inspection data formatted for university documentation, CSV spreadsheets, and machine-readable JSON.")

    if st.session_state.analysis_result is None:
        st.warning("Please analyze an image before generating export reports.")
    else:
        res = st.session_state.analysis_result
        meta = st.session_state.current_image_meta

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)

        # 1. CSV Report
        with col_r1:
            csv_data = ReportGenerator.generate_csv(res["coins"])
            st.download_button(
                label="📄 Download CSV Report",
                data=csv_data,
                file_name=f"coin_report_{meta.get('file_name', 'analysis')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # 2. JSON Report
        with col_r2:
            json_data = ReportGenerator.generate_json(
                analysis_result=res,
                image_metadata=meta,
                config_parameters={
                    "min_diam_px": min_diam,
                    "max_diam_px": max_diam,
                    "sensitivity": sensitivity,
                    "calibration_scale": calibration_scale
                }
            )
            st.download_button(
                label="📦 Download JSON Data",
                data=json_data,
                file_name=f"coin_analysis_{meta.get('file_name', 'analysis')}.json",
                mime="application/json",
                use_container_width=True
            )

        # 3. PDF Report
        with col_r3:
            try:
                pdf_bytes = ReportGenerator.generate_pdf(
                    analysis_result=res,
                    image_metadata=meta,
                    annotated_rgb=res["annotated_rgb"]
                )
                st.download_button(
                    label="📑 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"coin_inspection_report_{meta.get('file_name', 'analysis')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation error: {e}")

        # 4. Annotated Image Download
        with col_r4:
            annotated_pil = Image.fromarray(res["annotated_rgb"])
            img_buf = io.BytesIO()
            annotated_pil.save(img_buf, format="PNG")
            st.download_button(
                label="🖼️ Download Annotated Image",
                data=img_buf.getvalue(),
                file_name=f"annotated_{meta.get('file_name', 'detection')}.png",
                mime="image/png",
                use_container_width=True
            )

        st.markdown("### 📊 Preview Generated CSV Data")
        csv_df = pd.read_csv(io.StringIO(csv_data))
        st.dataframe(csv_df, use_container_width=True)


# ==========================================
# VIEW 8: ABOUT THE PROJECT
# ==========================================
elif nav_choice == "8. About the Project":
    st.title("ℹ️ About the Smart Coin Detection System")
    st.markdown(r"""
    ### Academic Computer Vision & Digital Image Processing Project
    **Course:** B.Sc. Computer Science / Digital Image Processing  
    **Topic:** Autonomous Coin Detection, Localization, Counting, Classification, and Condition Screening.

    ---

    ### 🛠️ Architecture & Implemented Techniques
    1. **Color Space Ingestion:** BGR to Luminance Grayscale conversion ($Y = 0.299R + 0.587G + 0.114B$), HSV color analysis, and CIELAB chromatic distance.
    2. **Noise Reduction:** 2D Gaussian Filtering ($\sigma=1.5$) to suppress sensor noise while preserving coin boundary gradients.
    3. **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Tile-based local histogram normalization preventing noise over-amplification in low-contrast scenes.
    4. **Boundary Extraction:** Dual-threshold Canny edge detection with $L_2$ gradient norm calculation.
    5. **Morphological Filtering:** Elliptical structuring element closing ($\bullet$) and opening ($\circ$) to seal perimeter gaps.
    6. **Touching / Overlapping Object Separation:** Euclidean Distance Transform ($L_2$ metric) coupled with Marker-Controlled Watershed segmentation.
    7. **Geometric Parametrization:** Contour area, arc perimeter, dimensionless circularity ($C = \frac{4\pi A}{P^2}$), and PCA orientation angles.
    8. **Multi-Modal Classification:** Calibrated millimeter diameter matching blended with LAB core/annular bimetallic color analysis against local JSON reference catalogs.
    9. **Surface Condition Screening:** High-pass scratch detection, tarnish delta-$E$ variance, and perimeter radial standard deviation.

    ---

    ### ⚠️ Limitations & Ethical Disclosures
    - **Perspective Distortion:** Planar image measurements strictly assume normal (perpendicular) camera alignment. Off-axis perspective tilts require projective homography rectification.
    - **Mass Determination:** 2D optical images cannot measure weight or physical mass. Official mint references and manual scale entries are clearly documented.
    - **Condition Screening:** Automated defect classification functions as a screening aid and does not constitute certified numismatic Sheldon Scale grading (e.g., MS-65).
    """)
