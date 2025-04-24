import streamlit as st
import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="IFCB ROI Viewer", layout="wide")
st.title("🔬 IFCB ROI Viewer")

# Ensure shared_state is initialized
if "shared_state" not in st.session_state:
    st.session_state.shared_state = {}

# Reset shared state defaults at start of each run
st.session_state.shared_state["results_df"] = None
st.session_state.shared_state["valid_roinums"] = []
st.session_state.shared_state["roi_metadata"] = []


tabs = st.tabs(["🔍 ROI Viewer", "📊 cnn Classification Summary"])

# ROI VIEWER TAB
with tabs[0]:
   folder = st.text_input("📂 Enter or paste full path to folder containing .roi files:", value="data")

    if os.path.isdir(folder):
        roi_files = sorted([f for f in os.listdir(folder) if f.endswith('.roi')])
        if not roi_files:
            st.warning("No .roi files found in selected folder.")
        else:
            selected_name = st.text_input("🔍 Search filename:")
            matching_files = [f for f in roi_files if selected_name.lower() in f.lower()]

            if matching_files:
                file_index = st.session_state.get("file_index", 0)
                if file_index >= len(matching_files):
                    file_index = 0

                col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                with col1:
                    if st.button("⏮️ First"):
                        file_index = 0
                with col2:
                    if st.button("⬅️ Previous"):
                        file_index = (file_index - 1) % len(matching_files)
                with col4:
                    if st.button("Next ➡️"):
                        file_index = (file_index + 1) % len(matching_files)
                with col5:
                    if st.button("⏭️ Last"):
                        file_index = len(matching_files) - 1

                st.session_state.file_index = file_index
                selected_file = matching_files[file_index]
                st.subheader(f"📄 Viewing: {selected_file}")

                base_name = os.path.splitext(selected_file)[0]
                bin_identifier = base_name.split("_", 1)[1]
                adc_path = os.path.join(folder, base_name + ".adc")
                hdr_path = os.path.join(folder, base_name + ".hdr")
                roi_path = os.path.join(folder, selected_file)

                results_prefix = base_name.replace("D", "CNN_D")
                results_filename = results_prefix + "_results.csv"
                results_path = os.path.join(folder, results_filename)
                if results_path and os.path.exists(results_path):
                    results_df = pd.read_csv(results_path)
                    results_df["roinum"] = results_df["roinum"].astype(int)
                else:
                    results_df = pd.DataFrame(columns=["roinum", "PredictedClassT", "classPT", "classP"])  # Empty but valid DataFrame
                    st.warning("⚠️ No CNN results file found for this bin — displaying images without classifications.")

                # Clear or update shared state with fresh data
                st.session_state.shared_state["selected_file"] = selected_file
                st.session_state.shared_state["results_df"] = results_df
                st.session_state.shared_state["valid_roinums"] = results_df["roinum"].unique().tolist() if results_df is not None else []

                def read_roi_images_with_metadata(roi_path, adc_path):
                    images = []
                    metadata = []
                    display_meta = []
                    try:
                        adc_df = pd.read_csv(adc_path, header=None)
                        start_bytes = adc_df.iloc[:, 17].values
                        widths = adc_df.iloc[:, 15].values
                        heights = adc_df.iloc[:, 16].values
                        times = adc_df.iloc[:, 11].values
                        peakA = adc_df.iloc[:, 6].values
                        peakB = adc_df.iloc[:, 7].values

                        for i in range(len(start_bytes)):
                            size = widths[i] * heights[i]
                            metadata.append({
                                "roi_index": i,
                                "offset": start_bytes[i],
                                "time": times[i],
                                "peakA": peakA[i],
                                "peakB": peakB[i],
                                "width": widths[i],
                                "height": heights[i],
                                "size": size
                            })

                        display_meta = sorted(metadata, key=lambda x: x["size"], reverse=True)

                        with open(roi_path, 'rb') as f:
                            for entry in display_meta:
                                offset = int(entry["offset"])
                                w = int(entry["width"])
                                h = int(entry["height"])
                                num_pixels = w * h
                                if offset < 0 or w <= 0 or h <= 0:
                                    images.append(None)
                                    continue
                                f.seek(offset)
                                img_bytes = f.read(num_pixels)
                                if len(img_bytes) < num_pixels:
                                    images.append(None)
                                    continue
                                img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w))
                                images.append(Image.fromarray(img_array))
                    except Exception as e:
                        st.error(f"Error reading ROI file: {e}")
                    return images, display_meta, metadata

                def read_hdr_metadata(hdr_path):
                    keys_of_interest = ["FileComment", "runTime", "triggerCount", "roiCount", "PMTAhighVoltage", "PMTBhighVoltage"]
                    metadata = {}
                    try:
                        with open(hdr_path, 'r') as f:
                            for line in f:
                                if ':' in line:
                                    key, value = line.strip().split(':', 1)
                                    if key.strip() in keys_of_interest:
                                        metadata[key.strip()] = value.strip()
                    except Exception as e:
                        st.warning(f"Unable to read .hdr file: {e}")
                    return metadata

                if os.path.exists(adc_path):
                    images, display_meta, cnn_meta = read_roi_images_with_metadata(roi_path, adc_path)

                    if os.path.exists(hdr_path):
                        with st.expander("🗂️ Bin Metadata from .hdr", expanded=True):
                            hdr_metadata = read_hdr_metadata(hdr_path)
                            st.markdown("<style> .hdr-key { font-weight: bold; } </style>", unsafe_allow_html=True)
                            for k, v in hdr_metadata.items():
                                st.markdown(f"- <span class='hdr-key'>{k}:</span> {v}", unsafe_allow_html=True)

                    if images:
                        filter_enabled = st.checkbox("✅ Enable CNN classPT filtering", value=False)
                        filter_pt = st.slider("🎚️ Filter by classPT (CNN confidence threshold):", min_value=0.01, max_value=1.0, value=0.01, step=0.01) if filter_enabled else 0.0
                        st.markdown("### 🦢 ROI Images (Sorted by Size)")
                        default_rois_per_page = 50
                        rois_per_page = st.number_input("📏 ROIs per page:", min_value=1, max_value=200, value=default_rois_per_page, step=1)

                        roi_page = st.session_state.get("roi_page", 0)
                        num_pages = int(np.ceil(len(images) / rois_per_page))

                        st.markdown(f"**Total ROIs in this file:** {len(images)}")

                        paginated_images = images[roi_page * rois_per_page:(roi_page + 1) * rois_per_page]
                        paginated_meta = display_meta[roi_page * rois_per_page:(roi_page + 1) * rois_per_page]

                        cols = st.columns(5)
                        for i, (img, meta) in enumerate(zip(paginated_images, paginated_meta)):
                            roi_num = meta["roi_index"] + 1
                            show_image = True
                            cnn_caption = "\nCNN: No classification"

                            if results_df is not None:
                                match = results_df[results_df["roinum"] == roi_num].copy()
                                if not match.empty:
                                    class_pt = match["classPT"].values[0]
                                    if not filter_enabled or class_pt >= filter_pt:
                                        pred_class = match["PredictedClassT"].values[0]
                                        class_p = match["classP"].values[0]
                                        cnn_caption = f"\nCNN: {pred_class} ({class_p:.2f})"
                                    else:
                                        show_image = False
                                else:
                                    show_image = not filter_enabled

                            if show_image and img is not None:
                                with cols[i % 5]:
                                    st.image(
                                        img,
                                        caption=(
                                            f"ROI {roi_num}\nTime: {meta['time']}\n"
                                            f"PeakA: {meta['peakA']:.2f}  PeakB: {meta['peakB']:.2f}\n"
                                            f"Size: {meta['width']}x{meta['height']}{cnn_caption}"
                                        ),
                                        use_container_width=True
                                    )

                        st.markdown(f"### File {file_index + 1} of {len(matching_files)}")
                        col_prev, col_next = st.columns(2)
                        with col_prev:
                            if st.button("⬅️ Prev Page") and roi_page > 0:
                                st.session_state["roi_page"] = roi_page - 1
                        with col_next:
                            if st.button("Next Page ➡️") and roi_page < num_pages - 1:
                                st.session_state["roi_page"] = roi_page + 1
                    else:
                        st.error("No images found in this .roi file.")
                else:
                    st.warning("Missing matching .adc file: " + base_name + ".adc")

                st.session_state.shared_state["roi_metadata"] = cnn_meta
            else:
                st.warning("No matching files found.")
    else:
        st.info("Please enter a valid folder path.")

# CLASSIFICATION SUMMARY TAB
with tabs[1]:
    st.header("📈 CNN Classification Summary")

    if "shared_state" not in st.session_state:
        st.session_state.shared_state = {}

    selected_file = st.session_state.shared_state.get("selected_file")
    results_df = st.session_state.shared_state.get("results_df")
    metadata = st.session_state.shared_state.get("roi_metadata")
    valid_roinums = st.session_state.shared_state.get("valid_roinums")

    if selected_file and results_df is not None and metadata:
        roi_order = [meta["roi_index"] + 1 for meta in metadata if (meta["roi_index"] + 1) in valid_roinums]
        results_df = results_df.set_index("roinum").loc[roi_order].reset_index()

        class_summary_df = results_df["PredictedClassT"].value_counts().reset_index()
        class_summary_df.columns = ["Class", "Count"]

        st.markdown(f"### Summary for: **{selected_file}**")
        st.dataframe(class_summary_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=class_summary_df, x="Class", y="Count", ax=ax)
        ax.set_title("Predicted Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)
    else:
        st.info("No CNN classification results available for this file.")
