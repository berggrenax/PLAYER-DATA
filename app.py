import streamlit as st

st.set_page_config(page_title="TEST", layout="wide")

st.title("Testar att Streamlit funkar i molnet")

st.write("Hej från molnet 👋")

st.write("Nycklar i st.secrets:")
st.write(list(st.secrets.keys()))
