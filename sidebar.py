import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("Telecom Churn App")
        st.markdown("### About")
        st.info(
            "This application is part of an **educational project** "
            "for learning Data Science and Machine Learning."
        )

        st.markdown("### Authors")
        st.markdown(
            """
            - Laura Fatta
            - Dora Novoa
            - Jaime Antonelli
            - Mehmet Erdem Sivri 
            """
        )