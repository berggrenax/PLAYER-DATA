import streamlit as st

st.set_page_config(page_title="PLAYER DATA", layout="wide")

# ---- Lösenord ----
APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)

def check_password():
    if not APP_PASSWORD:
        st.error("APP_PASSWORD saknas i secrets")
        st.stop()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return True

    pw = st.text_input("Ange lösenord:", type="password")
    if st.button("Logga in"):
        if pw == APP_PASSWORD:
            st.session_state.logged_in = True
            return True
        else:
            st.error("Fel lösenord")

    return False

if not check_password():
    st.stop()

st.title("PLAYER DATA")
st.write("Lösenord OK — appen fungerar ✔️")
