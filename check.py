import streamlit as st
import tensorflow as tf

def main():
    # Display the TensorFlow version
    st.write("TensorFlow version:", tf.__version__)

if __name__ == "__main__":
    main()
