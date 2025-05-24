import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="Rice Pest Detection",
    page_icon="🌾",
    layout="wide"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_trained_model():
    """Load the trained model with error handling"""
    MODEL_PATH = 'model.h5'  # Updated model name
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found. Please make sure the model file is in the same directory as this script.")
        return None
    
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Define class names based on your model training 
class_names = [
    'Bacterial leaf blight',
    'Brown spot', 
    'Fresh Leaves',
    'Leaf smut',
    'Planthopper',
    'Rice hispa',
    'Steam borer whiteheads'
]

def load_and_prep_image(img, img_size=128):
    """Preprocess the uploaded image"""
    try:
        # Convert to RGB if needed (handles different image formats)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((img_size, img_size))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_with_confidence(model, img_array, class_names, confidence_threshold=0.60):
    """Make prediction with confidence threshold"""
    try:
        # Get model predictions
        predictions = model.predict(img_array)
        # Get the highest confidence score
        max_confidence = np.max(predictions)
        # Get the index of the class with highest confidence
        predicted_class_index = np.argmax(predictions)
        
        # Get all prediction probabilities
        prediction_details = {}
        for i, class_name in enumerate(class_names):
            prediction_details[class_name] = predictions[0][i]
        
        # Check if confidence meets threshold
        if max_confidence >= confidence_threshold:
            predicted_class = class_names[predicted_class_index]
            return predicted_class, max_confidence, prediction_details
        else:
            return "Unknown/Not a rice leaf", max_confidence, prediction_details
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def create_confidence_chart(prediction_details):
    """Create a bar chart showing confidence for all classes"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(prediction_details.keys())
    probabilities = [prob * 100 for prob in prediction_details.values()]  # Convert to percentage
    
    # Create bar chart
    bars = ax.bar(classes, probabilities, color='lightblue', edgecolor='navy')
    
    # Highlight the highest probability
    max_prob_index = probabilities.index(max(probabilities))
    bars[max_prob_index].set_color('red')
    
    ax.set_title('Prediction Confidence for All Classes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Confidence (%)')
    ax.set_xlabel('Classes')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{prob:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Main Streamlit app
def main():
    # Header
    st.title("🌾 Rice Pest Detection System")
    st.markdown("Rice Plant Disease and Pest Classification")
    st.write("Upload an image of a rice plant leaf, and the model will classify whether it is disease or pest attack.")
    
    # Load model
    model = load_trained_model()
    if model is None:
        st.stop()  # Stop execution if model can't be loaded
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.6, 
        step=0.05,
        help="Lower values make the model more accepting, higher values make it more selective"
    )
    
    show_details = st.sidebar.checkbox("Show detailed probabilities", value=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a rice leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button('🔍 Analyze Image', type="primary"):
                with st.spinner('🔄 Processing image...'):
                    # Preprocess image
                    img_array = load_and_prep_image(img)
                    
                    if img_array is not None:
                        # Make prediction
                        predicted_class, confidence, prediction_details = predict_with_confidence(
                            model, img_array, class_names, confidence_threshold
                        )
                        
                        if predicted_class is not None:
                            # Store results in session state for display in col2
                            st.session_state.prediction_results = {
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'prediction_details': prediction_details
                            }
                            st.rerun()  # Refresh to show results
    
    with col2:
        st.subheader("📊 Results")
        
        # Check if we have prediction results
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            predicted_class = results['predicted_class']
            confidence = results['confidence']
            prediction_details = results['prediction_details']
            
            # Display main result
            if predicted_class == "Unknown/Not a rice leaf":
                st.error(f"🚫 **{predicted_class}**")
                st.write(f"**Confidence:** {confidence:.1%}")
                st.write("This doesn't appear to be a rice leaf .")
            elif predicted_class == "Fresh Leaves":
                st.success(f"✅ **{predicted_class}**")
                st.write(f"**Confidence:** {confidence:.1%}")
                st.write("Great! This appears to be a healthy rice leaf.")
            else:
                st.warning(f"⚠️ **{predicted_class}**")
                st.write(f"**Confidence:** {confidence:.1%}")
                st.write("This rice leaf shows signs of disease or pest attack.")
            
            # Show detailed probabilities if requested
            if show_details and prediction_details:
                st.subheader("📈 Detailed Analysis")
                
                # Create and display confidence chart
                fig = create_confidence_chart(prediction_details)
                st.pyplot(fig)
                
                # Show probability table
                st.subheader("📋 Probability Breakdown")
                prob_data = []
                for class_name, prob in prediction_details.items():
                    prob_data.append({
                        'Class': class_name,
                        'Probability': f"{prob:.1%}",
                        'Confidence': prob
                    })
                
                # Sort by confidence
                prob_data.sort(key=lambda x: x['Confidence'], reverse=True)
                
                # Display as table
                for item in prob_data:
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.write(item['Class'])
                    with cols[1]:
                        st.write(item['Probability'])
        
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results here.")
    
    
if __name__ == "__main__":
    main()