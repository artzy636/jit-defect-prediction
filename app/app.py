import streamlit as st

from src.config import SEED

st.title("JIT Defect Prediction Demo")
st.write("Just-in-time defect prediction on the ApacheJIT dataset.")
st.write(f"Random seed: {SEED}")
