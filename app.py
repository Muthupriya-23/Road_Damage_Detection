
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Road Damage Detection System")
st.write(
    "Upload a road image to detect road damage "
    "using a trained YOLO deep learning model."
)

# ---------------- MODEL PATH ----------------
MODEL_DIR = Path(__file__).parent / "models"
model_files = list(MODEL_DIR.glob("*.pt"))


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


if not model_files:
    st.error(
        "No .pt model found in the models folder. "
        "Place your downloaded model inside models/."
    )
    st.stop()

# Use the first .pt file found in the models folder
model_path = model_files[0]

try:
    model = load_model(str(model_path))
    st.success(f"Model loaded: {model_path.name}")
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)


# ---------------- DAMAGE ANALYSIS ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("Uploaded Road Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Road Damage", type="primary"):
        with st.spinner("Analysing road image..."):
            try:
                results = model.predict(
                    source=image_array,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]
                boxes = result.boxes

                # Display the annotated image
                annotated_bgr = result.plot()
                annotated_rgb = annotated_bgr[:, :, ::-1]

                st.subheader("Detection Result")
                st.image(
                    annotated_rgb,
                    caption="AI Detection Output",
                    use_container_width=True
                )

                if boxes is None or len(boxes) == 0:
                    st.warning(
                        "No damage detected. This does not guarantee "
                        "that the road is damage-free."
                    )
                else:
                    st.subheader("Damage Details")

                    names = result.names
                    image_height, image_width = image_array.shape[:2]
                    image_area = image_height * image_width

                    rows = []

                    for box in boxes:
                        class_id = int(box.cls.item())
                        label = names[class_id]
                        score = float(box.conf.item())

                        x1, y1, x2, y2 = (
                            box.xyxy[0].cpu().numpy()
                        )

                        box_area = max(
                            0, x2 - x1
                        ) * max(0, y2 - y1)

                        area_percent = (
                            box_area / image_area * 100
                            if image_area else 0
                        )

                        # Rough visual size estimate only
                        if area_percent < 1:
                            severity = "Low"
                        elif area_percent < 5:
                            severity = "Medium"
                        else:
                            severity = "High"

                        rows.append({
                            "Damage Type": str(label),
                            "Confidence": f"{score:.1%}",
                            "Estimated Severity": severity,
                            "Image Area Covered": f"{area_percent:.2f}%"
                        })

                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Severity is a rough estimate based on the "
                        "detected bounding-box size in the image. "
                        "It is not a validated measure of actual "
                        "road damage severity."
                    )

                    st.metric("Detected Objects", len(rows))

                    st.download_button(
                        "Download Detection Report (CSV)",
                        data=__import__("pandas").DataFrame(
                            rows
                        ).to_csv(index=False).encode("utf-8"),
                        file_name="road_damage_report.csv",
                        mime="text/csv"
                    )

            except Exception as error:
                st.error(f"Detection failed: {error}")
