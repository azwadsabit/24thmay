import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import requests  # For ESP32 communication

# ESP32 Configuration
ESP32_IP = "192.168.71.176"

# Set Streamlit page config
st.set_page_config(page_title="Rice Pest Detection", page_icon="🌾", layout="wide")

# Cache the model loading
@st.cache_resource
def load_trained_model():
    MODEL_PATH = 'model.h5'
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Class names
class_names = [
    'Bacterial leaf blight',
    'Brown spot',
    'Fresh Leaves',
    'Leaf smut',
    'Planthopper',
    'Rice hispa',
    'Steam borer whiteheads'
]

# Pest-specific frequencies (in Hz)
pest_frequencies = {
    'Planthopper': 40000,
    'Rice hispa': 35000,
    'Steam borer whiteheads': 45000
}

def load_and_prep_image(img, img_size=128):
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((img_size, img_size))
        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def predict_with_confidence(model, img_array, class_names, confidence_threshold=0.60):
    try:
        predictions = model.predict(img_array)
        max_conf = np.max(predictions)
        pred_idx = np.argmax(predictions)

        pred_details = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

        if max_conf >= confidence_threshold:
            return class_names[pred_idx], max_conf, pred_details
        else:
            return "Unknown/Not a rice leaf", max_conf, pred_details
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def create_confidence_chart(prediction_details):
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(prediction_details.keys())
    probs = [prob * 100 for prob in prediction_details.values()]
    bars = ax.bar(classes, probs, color='lightblue', edgecolor='navy')
    bars[np.argmax(probs)].set_color('red')
    ax.set_title("Prediction Confidence", fontsize=14)
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Class")
    plt.xticks(rotation=45)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{prob:.1f}%', ha='center')
    plt.tight_layout()
    return fig

def send_command_to_esp32(action, freq=None):
    try:
        if action == "on":
            url = f"http://{ESP32_IP}/on"
        elif action == "off":
            url = f"http://{ESP32_IP}/off"
        elif action == "freq" and freq:
            url = f"http://{ESP32_IP}/freq?value={freq}"
        else:
            st.warning("Invalid command or missing frequency.")
            return

        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            st.success(f"Command sent successfully!")
        else:
            st.error(f"Failed to send command: {response.status_code}")
    except Exception as e:
        st.error(f"ESP32 communication error: {e}")

# Streamlit App
def main():
    st.title("🌾 Rice Pest Detection System")
    st.write("Upload a rice leaf image and this model will identify if it's pest attack or diseases")

    model = load_trained_model()
    if model is None:
        st.stop()

    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
    show_details = st.sidebar.checkbox("Show Confidence Chart", value=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Analyze Image"):
                with st.spinner("Processing..."):
                    img_array = load_and_prep_image(img)
                    if img_array is not None:
                        predicted_class, confidence, prediction_details = predict_with_confidence(
                            model, img_array, class_names, confidence_threshold
                        )
                        if predicted_class:
                            st.session_state.prediction_results = {
                                "class": predicted_class,
                                "confidence": confidence,
                                "details": prediction_details
                            }
                            st.rerun()

    with col2:
        st.subheader("📊 Detection Result")

        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            st.write(f"**Class:** `{results['class']}`")
            st.write(f"**Confidence:** `{results['confidence']:.2%}`")

            if results["class"] == "Fresh Leaves":
                st.success("✅ Healthy leaf detected.")
            elif results["class"] == "Unknown/Not a rice leaf":
                st.error("🚫 Unable to recognize rice leaf.")
            else:
                st.warning("⚠️ Pest or disease detected!")

            if show_details:
                fig = create_confidence_chart(results["details"])
                st.pyplot(fig)

    st.markdown("---")
    st.subheader("🎛 Pest Repeller Control")
    
    # Display detected pest and suggested frequency
    if 'prediction_results' in st.session_state:
        detected_pest = st.session_state.prediction_results["class"]
        if detected_pest in pest_frequencies:
            st.info(f"🎯 **Detected Pest:** {detected_pest}")
            st.info(f"🔊 **Recommended Frequency:** {pest_frequencies[detected_pest]:,} Hz")
    
    col_on, col_off = st.columns([1, 1])

    with col_on:
        if st.button("🟢 Turn ON Repeller"):
            send_command_to_esp32("on")
    
    with col_off:
        if st.button("🔴 Turn OFF Repeller"):
            send_command_to_esp32("off")

    st.markdown("### 🎚 Frequency Control")
    
    # Quick frequency buttons for detected pests
    if 'prediction_results' in st.session_state:
        detected_pest = st.session_state.prediction_results["class"]
        if detected_pest in pest_frequencies:
            if st.button(f"🎯 Set Frequency for {detected_pest} ({pest_frequencies[detected_pest]:,} Hz)"):
                send_command_to_esp32("freq", pest_frequencies[detected_pest])

    # Manual frequency control
    col_freq1, col_freq2, col_freq3 = st.columns(3)
    
    with col_freq1:
        if st.button("🦗 Planthopper (40 kHz)"):
            send_command_to_esp32("freq", 40000)
    
    with col_freq2:
        if st.button("🐛 Rice Hispa (35 kHz)"):
            send_command_to_esp32("freq", 35000)
    
    with col_freq3:
        if st.button("🌾 Stem Borer (45 kHz)"):
            send_command_to_esp32("freq", 45000)

    # Custom frequency input
    st.markdown("### 🔧 Custom Frequency")
    custom_freq = st.number_input("Set Custom Frequency (Hz)", min_value=1000, max_value=50000, value=25000, step=1000)
    if st.button("🎚 Set Custom Frequency"):
        send_command_to_esp32("freq", custom_freq)

    st.markdown("---")
    st.info("💡 **Instructions:** First analyze the image to detect pests, then manually turn ON the repeller and set the appropriate frequency.")

if __name__ == "__main__":
    main()
