# import streamlit as st

# st.title("👨‍⚕️ Doctor Login")

# username = st.text_input("Username")
# password = st.text_input("Password", type="password")

# if st.button("Login"):
#     if username == "doctor" and password == "admin123":
#         st.success("Login successful")
#         st.session_state["doctor_logged_in"] = True
#         st.switch_page("pages/2_Doctor_Panel.py")
#     else:
#         st.error("Invalid credentials")

import streamlit as st

st.title("👨‍⚕️ Doctor Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username == "doctor" and password == "admin123":
        st.success("Login successful")
        st.session_state["doctor_logged_in"] = True
        st.switch_page("pages/2_Doctor_Panel.py")
    else:
        st.error("Invalid credentials")
