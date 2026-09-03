import torch
import streamlit as st
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from torchvision import transforms

from models.attention_model import CornDiseaseAttentionNet
from src.gradcam import GradCAM


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Corn Leaf Disease Detection",
    page_icon="🌽",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f4f7f5;
        margin-top: 15px;
    }

    .metric-title {
        font-size: 17px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌽 Corn Leaf Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Attention-based Deep Learning System for Corn Leaf Disease Classification
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy"
]


# =========================================================
# DISEASE INFORMATION
# =========================================================

disease_info = {

    "Blight": {
        "name": "Blight",
        "status": "Disease Detected",
        "description":
            "Blight may cause elongated brown or tan lesions and damaged leaf tissue.",
        "recommendation":
            "Inspect nearby plants, remove heavily affected leaves where appropriate, "
            "and consult an agricultural expert for suitable disease-management options."
    },

    "Common_Rust": {
        "name": "Common Rust",
        "status": "Disease Detected",
        "description":
            "Common Rust generally produces small reddish-brown pustules on corn leaves.",
        "recommendation":
            "Monitor disease spread and consult local agricultural guidance for "
            "appropriate resistant varieties or treatment options."
    },

    "Gray_Leaf_Spot": {
        "name": "Gray Leaf Spot",
        "status": "Disease Detected",
        "description":
            "Gray Leaf Spot commonly appears as elongated gray or brown lesions on leaves.",
        "recommendation":
            "Monitor humidity and crop conditions, reduce infected residue where appropriate, "
            "and seek expert advice for disease management."
    },

    "Healthy": {
        "name": "Healthy",
        "status": "No Disease Detected",
        "description":
            "The uploaded leaf does not show strong visual features associated with the "
            "three disease classes recognized by this model.",
        "recommendation":
            "Continue regular crop monitoring and maintain good field-management practices."
    }
}


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = CornDiseaseAttentionNet(
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(
        torch.load(
            "models/best_corn_attention_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model


model = load_model()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Project Information")

    st.write("**Model:** CNN + Attention")

    st.write("**Input Size:** 224 × 224")

    st.write("**Classes:** 4")

    st.write("**Framework:** PyTorch")

    st.write("**Interface:** Streamlit")

    st.divider()

    st.write("### Supported Classes")

    st.write("• Blight")
    st.write("• Common Rust")
    st.write("• Gray Leaf Spot")
    st.write("• Healthy")

    st.divider()

    st.caption(
        "This system is an academic AI project and should not replace professional agricultural diagnosis."
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a Corn Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -----------------------------------------------------
    # PREPARE INPUT
    # -----------------------------------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)


    # -----------------------------------------------------
    # NORMAL PREDICTION
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )


    predicted_class = class_names[
        predicted.item()
    ]

    confidence_score = (
        confidence.item() * 100
    )


    # -----------------------------------------------------
    # GRAD-CAM
    # -----------------------------------------------------

    gradcam = GradCAM(
        model,
        model.features[-2]
    )

    cam = gradcam.generate(
        input_tensor,
        predicted.item()
    )


    original_np = np.array(
        image.resize((224, 224))
    )


    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )


    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    overlay = (
        0.6 * original_np
        +
        0.4 * heatmap
    ).astype(np.uint8)


    # =====================================================
    # DISPLAY IMAGE + RESULT
    # =====================================================

    left_col, right_col = st.columns(2)


    with left_col:

        st.subheader("Uploaded Leaf")

        st.image(
            image,
            use_container_width=True
        )


    with right_col:

        st.subheader("Prediction Result")

        if predicted_class == "Healthy":

            st.success(
                f"🌿 {disease_info[predicted_class]['status']}"
            )

        else:

            st.warning(
                f"⚠️ {disease_info[predicted_class]['status']}"
            )


        st.metric(
            label="Predicted Class",
            value=disease_info[predicted_class]["name"]
        )


        st.metric(
            label="Confidence",
            value=f"{confidence_score:.2f}%"
        )


        st.write(
            disease_info[predicted_class]["description"]
        )


    st.divider()


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    st.subheader("Prediction Probabilities")

    probs = probabilities[
        0
    ].detach().cpu().numpy()


    probability_data = {

        "Disease Class": [
            "Blight",
            "Common Rust",
            "Gray Leaf Spot",
            "Healthy"
        ],

        "Probability (%)": [
            round(float(p) * 100, 2)
            for p in probs
        ]
    }


    st.bar_chart(
        data={
            class_names[i]:
            float(probs[i]) * 100

            for i in range(
                len(class_names)
            )
        }
    )


    for class_name, probability in zip(
        class_names,
        probs
    ):

        st.write(
            f"**{class_name.replace('_', ' ')}:** "
            f"{probability * 100:.2f}%"
        )


    st.divider()


    # =====================================================
    # ATTENTION HEATMAP
    # =====================================================

    st.subheader("Attention Heatmap")

    st.write(
        "The highlighted regions indicate areas that influenced the model's prediction."
    )


    heat_col1, heat_col2 = st.columns(2)


    with heat_col1:

        st.image(
            original_np,
            caption="Original Image",
            use_container_width=True
        )


    with heat_col2:

        st.image(
            overlay,
            caption="Grad-CAM Visualization",
            use_container_width=True
        )


    st.divider()


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.subheader("Recommendation")

    st.info(
        disease_info[
            predicted_class
        ]["recommendation"]
    )


    st.divider()


    # =====================================================
    # TECHNICAL DETAILS
    # =====================================================

    with st.expander(
        "View Technical Details"
    ):

        st.write(
            "**Model Architecture:** Custom CNN + Attention"
        )

        st.write(
            "**Image Resolution:** 224 × 224"
        )

        st.write(
            f"**Inference Device:** {device}"
        )

        st.write(
            f"**Predicted Class Index:** {predicted.item()}"
        )

        st.write(
            f"**Confidence:** {confidence_score:.4f}%"
        )


else:

    st.info(
        "👆 Upload a JPG or PNG image of a corn leaf to begin prediction."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Corn Leaf Disease Detection using Attention Network | "
    "Deep Learning & Computer Vision Project"
)