import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
import google.generativeai as genai
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure genai with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set to CPU only for custom CNN model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

output_dir = "saliency_maps"
os.makedirs(output_dir, exist_ok=True)

def generate_explanation(img_path, model_prediction, confidence):
    prompt = f"""
You are a distinguished neurologist and MRI diagnostic expert, recognized globally for your expertise in brain tumor detection and radiological interpretation. You have been called upon to provide a comprehensive analysis of a saliency map produced by a cutting-edge deep learning model. This model has been rigorously trained to classify MRI brain scans into one of four categories: glioma, meningioma, pituitary tumor, or no tumor.

The saliency map highlights regions of interest in the MRI scan, particularly those areas marked in light cyan, which the model considers most critical for its classification. For this specific MRI scan, the model has classified it as '{model_prediction}' with a confidence level of {confidence * 100}%. Your task is to offer an expert-level interpretation of these highlighted areas and the model's decision.

Your response must include detailed insights into the following aspects:

1. **Identification of Critical Anatomical Regions**: 
    - Identify the specific anatomical regions of the brain that are emphasized in the saliency map.
    - Provide precise descriptions of these regions in the context of their typical MRI appearance and their role in the model's classification.

2. **Correlation with the Predicted Tumor Type ('{model_prediction}')**: 
    - Explain how the highlighted areas correlate with the presence or absence of the predicted condition.
    - Discuss the known biological or structural changes in these regions that are characteristic of '{model_prediction}'.

3. **Saliency Map Patterns and Interpretations**: 
    - Examine the distribution and intensity of the cyan highlights. Are there patterns or clusters that suggest key areas of the model's focus? 
    - Interpret how these patterns align with known radiological features of glioma, meningioma, pituitary tumor, or normal brain anatomy.

4. **Biological and Clinical Significance**:
    - Delve into the biological reasons why these specific regions might be significant for diagnosing the '{model_prediction}'.
    - For tumor categories, discuss typical growth patterns, common regions of origin, and expected effects on nearby structures visible in MRI scans.

5. **Model Decision Validation**:
    - Critically assess how well the model’s highlighted regions support its prediction. 
    - Discuss whether the saliency map's focus is consistent with established clinical knowledge of '{model_prediction}'.
    - If the prediction is “no tumor,” explain why the saliency map avoids focusing on tumor-indicative regions.

6. **Potential Limitations or Anomalies**:
    - Identify any potential concerns with the saliency map. Are there regions of focus that seem clinically irrelevant or contradictory to the prediction?
    - Offer a hypothesis for any anomalies or unexpected focus areas.

7. **Implications for Clinical Decision-Making**:
    - Reflect on how this analysis might influence the patient’s clinical pathway. Could these findings suggest a need for additional scans, biopsy, or alternative treatment plans?

8. **Technical and Clinical Fusion**:
    - Bridge the technical aspects of the saliency map with your clinical expertise. Offer a synthesis that validates the model's decision-making while providing a clinically actionable perspective.

9. **Depth of Response**:
    - Avoid redundancy and ensure each sentence contributes a new layer of insight. 
    - Provide concrete examples or references to clinical scenarios, if applicable, to reinforce your interpretations.
    - Structure your response with clear sections or paragraphs to maintain logical flow and coherence.

Your analysis should aim for clarity, depth, and precision, with a length of 10–12 sentences or more if required to comprehensively address all aspects. The goal is to produce a detailed, authoritative interpretation that harmonizes advanced AI insights with deep clinical understanding.

Let’s proceed methodically, step by step, as we decode the map with surgical precision.
"""
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def generate_saliency_map(model, img_array, class_index, img_size):
    """Generate a saliency map to highlight areas of importance for the model's prediction."""
    try:
        with tf.GradientTape() as tape:
            img_tensor = tf.convert_to_tensor(img_array)
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            target_class = predictions[:, class_index]

        # Compute gradients
        gradients = tape.gradient(target_class, img_tensor)
        gradients = tf.math.abs(gradients)  # Absolute values of gradients
        gradients = tf.reduce_max(gradients, axis=-1).numpy().squeeze()

        # Normalize gradients
        gradients = np.clip(gradients, 0, None)
        if gradients.max() > 0:
            gradients /= gradients.max()

        # Resize gradients to match the image size
        gradients_resized = cv2.resize(gradients, img_size)

        # Apply a heatmap color map
        heatmap = cv2.applyColorMap(np.uint8(255 * gradients_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Create a superimposed image
        original_img = (img_array[0] * 255).astype("uint8")
        superimposed_img = cv2.addWeighted(heatmap, 0.6, original_img, 0.4, 0)

        # Save the saliency map
        saliency_map_path = os.path.join(output_dir, "saliency_map.jpg")
        cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

        return saliency_map_path
    except Exception as e:
        st.error(f"Error generating saliency map: {e}")
        return None

def load_xception_model(model_path):
    img_shape = (150, 150, 3)
    base_model = Xception(include_top=False, weights=None, input_shape=img_shape, pooling='max')

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation='relu', name='dense'),
        Dropout(rate=0.25),
        Dense(4, activation='softmax', name='dense_1')
    ])

    model.build(input_shape=(None,) + img_shape)
    model.load_weights(model_path)
    return model

def load_custom_cnn_model(model_path):
    model = load_model(model_path)
    return model

st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI scan for AI-driven classification and analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    selected_model = st.radio(
        "Select Model",
        ("Transfer Learning - Xception", "Custom CNN")
    )

    if selected_model == "Transfer Learning - Xception":
        model = load_xception_model("xception_model.weights.h5")
        img_size = (150, 150)
    else:
        model = load_custom_cnn_model("cnn_model.h5")
        img_size = (224, 224)

    labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    result = labels[class_index]
    confidence = prediction[0][class_index]

    st.markdown(f"""
        <style>
            .result-card {{
                background: linear-gradient(145deg, #1e1e1e, #333);
                padding: 20px;
                border-radius: 15px;
                color: white;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            .result-card:hover {{
                transform: scale(1.05);
            }}
            .confidence-bar {{
                background: #333;
                border-radius: 10px;
                overflow: hidden;
                margin-top: 10px;
                height: 15px;
            }}
            .confidence-fill {{
                height: 100%;
                background: linear-gradient(90deg, #4caf50, #8bc34a);
                width: {confidence*100}%;
                transition: width 0.6s ease;
            }}
            .result-text {{
                font-size: 1.5em;
                font-weight: bold;
            }}
            .confidence-text {{
                font-size: 1.2em;
                color: #FFD700;
            }}
        </style>

        <div class="result-card">
            <div class="result-text">Prediction: {result}</div>
            <div class="confidence-text">Confidence: {confidence:.2%}</div>
            <div class="confidence-bar">
                <div class="confidence-fill"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    saliency_map_path = generate_saliency_map(model, img_array, class_index, img_size)
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)
    with col2:
        st.image(saliency_map_path, caption='Model-Generated Saliency Map', use_container_width=True)

    st.write("## Class Probability Breakdown")
    probabilities = prediction[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probabilities = probabilities[sorted_indices]

    fig = go.Figure(go.Bar(
        x=sorted_probabilities,
        y=sorted_labels,
        orientation='h',
        marker=dict(
            color=sorted_probabilities,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Probability",
                tickformat=".0%",
                titlefont=dict(color='white')
            )
        ),
        text=[f"{p:.2%}" for p in sorted_probabilities],
        textposition="inside",
        insidetextanchor="middle"
    ))

    fig.update_layout(
        title='Class Probability Breakdown',
        xaxis=dict(title='Probability', tickformat=".0%", showgrid=False, color='white'),
        yaxis=dict(title='Class', showgrid=False, autorange="reversed", color='white'),
        title_font=dict(size=20, color='white'),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        height=450,
        width=700,
    )

    st.plotly_chart(fig)

    explanation = generate_explanation(saliency_map_path, result, confidence)
    st.write("## Expert Analysis of Saliency Map")
    st.write(explanation)

    del model
    gc.collect()
    tf.keras.backend.clear_session()
