import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from keras.models import load_model

# Load model
MODEL = load_model("best_model.keras")  # Make sure this file exists

# Label map
LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

# Initialize session state
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# Streamlit App
st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0-9) below and click 'Predict'.")

# Create canvas component with unique key
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)

# Button columns
col1, col2 = st.columns(2)

# Predict button
with col1:
    if st.button("🔮 Predict"):
        if canvas_result.image_data is not None:
            # Preprocess image
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            img = img / 255.0

            # Make prediction
            prediction = MODEL.predict(img)
            predicted_label = LABELS[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Display prediction
            st.success(f"**Prediction:** {predicted_label} \n\n**Confidence:** {confidence:.2f}%")
        else:
            st.warning("Please draw a digit first!")

# Clear button - Proper implementation
with col2:
    if st.button("🧹 Clear Canvas"):
        # Increment key to force canvas reset
        st.session_state.canvas_key += 1
        st.rerun()

# Optional: Add some styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        padding: 10px;
        border-radius: 5px;
    }
    .st-cn {
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)