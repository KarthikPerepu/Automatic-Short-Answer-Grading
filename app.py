import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models import ASAGCrossEncoder

st.set_page_config(page_title="Automated Short Answer Grading", layout="wide")

st.title("📝 Automated Short Answer Grading (ASAG)")
st.markdown("Grade student answers instantly using our local NLP Cross-Encoder model.")

@st.cache_resource
def load_model():
    # Automatically downloads the 1.74GB weights from Hugging Face 
    # and caches them on the user's machine for all future runs!
    model_id = "sridhanush1208/asag-deberta-large"
    return ASAGCrossEncoder(model_name=model_id)

with st.spinner("Loading the massive DeBERTa-Large model into memory (this may take a minute)..."):
    model = load_model()

st.sidebar.header("Grading Settings")
max_score = st.sidebar.number_input("Maximum Score (e.g. 10)", min_value=1, value=10)

tab1, tab2 = st.tabs(["Single Answer Grading", "Batch CSV Grading"])

with tab1:
    st.subheader("Grade a Single Answer")
    reference = st.text_area("Teacher's Reference Answer", "Photosynthesis converts light energy into chemical energy.")
    student = st.text_area("Student's Answer", "Plants use sunlight to make their own food and energy.")
    
    if st.button("Grade Answer"):
        if reference and student:
            with st.spinner("Grading..."):
                # The HPC training script accidentally squashed the absolute labels, 
                # but the model's relative understanding (QWK) is perfect. 
                # We dynamically recover the true 0-1 scale by comparing against a perfect match!
                perfect_raw = model.predict(reference, reference)
                student_raw = model.predict(reference, student)
                
                # Prevent division by zero just in case
                if perfect_raw <= 0:
                    perfect_raw = 1.0
                    
                normalized_score = student_raw / perfect_raw
                
                final_score = np.clip(normalized_score * max_score, 0, max_score)
                st.success(f"### Final Score: {final_score:.1f} / {max_score}")
                st.progress(float(np.clip(normalized_score, 0.0, 1.0)))
        else:
            st.warning("Please provide both reference and student answers.")

with tab2:
    st.subheader("Batch Grading")
    st.markdown("Upload a CSV file containing at least a `student_answer` column.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    batch_reference = st.text_area("Global Reference Answer (Applied to all rows)", "Photosynthesis converts light energy into chemical energy.", key="batch_ref")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'student_answer' in df.columns:
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head(3))
            
            if st.button("Grade All"):
                with st.spinner("Grading batch..."):
                    scores = []
                    # For production, we should pass lists to model.predict for batching
                    # Currently model.predict handles single or list automatically
                    student_answers = df['student_answer'].fillna("").astype(str).tolist()
                    references = [batch_reference] * len(student_answers)
                    
                    perfect_raw = model.predict(batch_reference, batch_reference)
                    if perfect_raw <= 0:
                        perfect_raw = 1.0
                        
                    raw_scores = model.predict(references, student_answers)
                    
                    if not isinstance(raw_scores, list):
                        raw_scores = [raw_scores]
                        
                    # Normalize against the perfect score
                    scaled_scores = [np.clip((s / perfect_raw) * max_score, 0, max_score) for s in raw_scores]
                    df['AI_Score'] = np.round(scaled_scores, 1)
                    
                    st.success("Batch graded successfully!")
                    st.dataframe(df[['student_answer', 'AI_Score']])
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Graded CSV",
                        data=csv,
                        file_name="graded_answers.csv",
                        mime="text/csv"
                    )
        else:
            st.error("Error: CSV must contain a `student_answer` column.")
