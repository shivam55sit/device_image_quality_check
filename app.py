import streamlit as st
import os
import zipfile
import shutil
import pandas as pd
import json
import time
from datetime import datetime
from PIL import Image

# Correct Import (assuming app.py is in the same folder as MainQualitycheck.py)
try:
    from MainQualitycheck import EyeQualityAssessment, OverallQuality
except ImportError as e:
    st.error(f"Failed to import from MainQualitycheck.py. Ensure the file exists in the same directory.\nError: {e}")
    st.stop()

# ---------------- CONFIG ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOOD_DIR = os.path.join(BASE_DIR, "good_tele")
DEFAULT_BAD_DIR = os.path.join(BASE_DIR, "bad_tele")
DEFAULT_REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Model Paths Configuration (Relative to app.py -> ../Grabi_chatbot/models)
MODELS_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'Grabi_chatbot', 'models'))

MODEL_CONFIG = {
    'eye_model_dir': os.path.join(MODELS_ROOT, 'peakmodels'),
    'use_eye_ensemble': True,
    'focus_model_path': os.path.join(MODELS_ROOT, 'focus_svm_model.joblib'),
    'focus_scaler_path': os.path.join(MODELS_ROOT, 'focus_scaler.joblib'),
    'focus_feature_names_path': os.path.join(MODELS_ROOT, 'focus_feature_names.txt'),
    'illumination_model_dir': MODELS_ROOT,
    'reflection_model_path': os.path.join(MODELS_ROOT, 'best_mobilevit_model.pth'),
    'completeness_model_path': os.path.join(MODELS_ROOT, 'resnet_completeness2.pth'),
    'completeness_xgb_model_path': os.path.join(MODELS_ROOT, 'xgboost_completeness2.json'),
    'resolution_model_path': os.path.join(MODELS_ROOT, 'resnet_resolution.pth'),
    'resolution_xgb_model_path': os.path.join(MODELS_ROOT, 'xgboost_resolution_model.pkl')
}

# ---------------- UI CONFIG ---------------- #
st.set_page_config(page_title="Tele-Eye QC Batch Processor", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .stat-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("👁️ Tele-Ophthalmology Image QC")
st.markdown("Batch assessment system for classifying eye images into **Good** or **Bad** quality.")

# ---------------- MODEL LOADER ---------------- #
@st.cache_resource
def load_quality_pipeline():
    """Load the specialized EyeQualityAssessment pipeline once."""
    try:
        pipeline = EyeQualityAssessment()
        # Verify model paths exist before loading
        if not os.path.exists(MODEL_CONFIG['eye_model_dir']):
             st.warning(f"⚠️ Model directory not found: {MODEL_CONFIG['eye_model_dir']}")
        
        pipeline.load_models(MODEL_CONFIG)
        return pipeline
    except Exception as e:
        return str(e)

with st.spinner("Loading AI Models... This may take a minute..."):
    pipeline_or_error = load_quality_pipeline()

if isinstance(pipeline_or_error, str):
    st.error(f"❌ Failed to load models: {pipeline_or_error}")
    st.info("Please check if the 'models' directory exists in the parent folder.")
    st.stop()
else:
    pipeline = pipeline_or_error
    st.success("✅ AI Diagnostics Engine Ready")

# ---------------- SIDEBAR CONFIG ---------------- #
st.sidebar.header("⚙️ Configuration")

input_method = st.sidebar.radio("Input Method", ["Local Folder Path", "Upload ZIP File"])

output_good = st.sidebar.text_input("Good Images Output Folder", value=DEFAULT_GOOD_DIR)
output_bad = st.sidebar.text_input("Bad Images Output Folder", value=DEFAULT_BAD_DIR)
output_reports = st.sidebar.text_input("Reports Output Folder", value=DEFAULT_REPORT_DIR)

# ---------------- MAIN LOGIC ---------------- #

def process_image(img_path, good_dir, bad_dir):
    """Run assessment on a single image and move it."""
    try:
        # Run Quality Check
        res = pipeline.mainqualitycheck(img_path)
        
        overall = res.get("overall_quality", "Error")
        filename = os.path.basename(img_path)
        
        # Determine Destination
        if overall == "Good Quality":
            target_folder = good_dir
            final_label = "GOOD"
        else:
            target_folder = bad_dir
            final_label = "BAD"
        
        # Copy Image
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy2(img_path, os.path.join(target_folder, filename))
        
        return {
            "filename": filename,
            "original_path": img_path,
            "overall_quality": overall,
            "final_label": final_label,
            "quality_pattern": res.get("quality_pattern", "N/A"),
            "recommendations": "; ".join(res.get("recommendations", [])),
            "processing_time": res.get("processing_time", 0)
        }
        
    except Exception as e:
        return {
            "filename": os.path.basename(img_path),
            "original_path": img_path,
            "overall_quality": "Error",
            "final_label": "ERROR",
            "error_details": str(e)
        }

def start_batch_processing(image_files, base_temp_dir=None):
    if not image_files:
        st.warning("No images found to process.")
        return

    st.divider()
    st.subheader(f"🚀 Processing {len(image_files)} Images")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    # Create directories if they don't exist
    os.makedirs(output_good, exist_ok=True)
    os.makedirs(output_bad, exist_ok=True)
    os.makedirs(output_reports, exist_ok=True)
    
    start_time = time.time()
    
    col1, col2, col3 = st.columns(3)
    metric_good = col1.empty()
    metric_bad = col2.empty()
    metric_processed = col3.empty()
    
    for idx, img_path in enumerate(image_files):
        # Update UI
        status_text.text(f"Analyzing: {os.path.basename(img_path)}...")
        
        # Process
        result = process_image(img_path, output_good, output_bad)
        results.append(result)
        
        # Update Metrics
        processed_count = idx + 1
        good_count = sum(1 for r in results if r['final_label'] == 'GOOD')
        bad_count = sum(1 for r in results if r['final_label'] == 'BAD')
        
        metric_processed.metric("Processed", f"{processed_count}/{len(image_files)}")
        metric_good.metric("✅ Good", good_count)
        metric_bad.metric("❌ Bad", bad_count)
        
        progress_bar.progress(processed_count / len(image_files))

    total_time = time.time() - start_time
    status_text.text(f"✅ Completed in {total_time:.2f} seconds!")
    progress_bar.progress(1.0)
    
    # Save Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    
    csv_filename = f"qc_report_{timestamp}.csv"
    csv_path = os.path.join(output_reports, csv_filename)
    df.to_csv(csv_path, index=False)
    
    st.success(f"Report saved to: {csv_path}")
    
    st.dataframe(df, use_container_width=True)
    
    # Cleanup temp dir if using zip
    if base_temp_dir and os.path.exists(base_temp_dir):
        try:
            shutil.rmtree(base_temp_dir)
        except Exception:
            pass # Best effort cleanup

# ---------------- INPUT HANDLERS ---------------- #

if input_method == "Local Folder Path":
    
    # Initialize session state for folder path if not present
    if "folder_path" not in st.session_state:
        st.session_state.folder_path = ""

    col_input, col_btn = st.columns([0.85, 0.15])
    
    with col_input:
        # Update text input value from session state
        folder_path = st.text_input(
            "Enter the full path to your image folder:", 
            value=st.session_state.folder_path,
            placeholder=r"C:\Users\Name\Desktop\images",
            key="folder_input_widget"
        )
    
    with col_btn:
        st.write("") # Spacer
        st.write("") # Spacer
    with col_btn:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("📂 Browse"):
            try:
                import subprocess
                import sys
                
                # Define a simple script to open the dialog
                # We run this in a separate process to avoid thread conflicts with Streamlit
                cmd = [
                    sys.executable, 
                    "-c", 
                    "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1); print(filedialog.askdirectory(master=root))"
                ]
                
                # Run the command and capture output
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Parse the output
                selected_folder = result.stdout.strip()
                
                if selected_folder:
                    st.session_state.folder_path = selected_folder
                    st.rerun()
            except Exception as e:
                st.error(f"Could not open file picker: {e}")
    
    # Sync manual text entry back to session state if it changes
    if folder_path != st.session_state.folder_path:
        st.session_state.folder_path = folder_path

    if st.button("Start Processing from Folder", type="primary"):
        if os.path.isdir(folder_path):
            candidates = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ]
            if candidates:
                start_batch_processing(candidates)
            else:
                st.error("No image files (.png, .jpg, .jpeg) found in that directory.")
        else:
            st.error("Invalid directory path. Please check the path and try again.")

elif input_method == "Upload ZIP File":
    uploaded_file = st.file_uploader("Upload a ZIP file containing images", type="zip")
    
    if uploaded_file and st.button("Start Processing ZIP"):
        # Create temp dir for extraction
        temp_dir = os.path.join(BASE_DIR, "temp_extract_" + datetime.now().strftime("%H%M%S"))
        os.makedirs(temp_dir, exist_ok=True)
        
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # Walk through temp dir to find images (recursive)
        image_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(os.path.join(root, file))
                    
        start_batch_processing(image_files, base_temp_dir=temp_dir)
