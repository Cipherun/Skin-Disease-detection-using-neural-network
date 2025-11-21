# import streamlit as st
# import sqlite3
# import pandas as pd

# if "doctor_logged_in" not in st.session_state:
#     st.error("You must log in.")
#     st.stop()

# st.title("📋 Doctor Dashboard - Patient Reports")

# conn = sqlite3.connect("patients.db")
# df = pd.read_sql_query("SELECT * FROM results ORDER BY timestamp DESC", conn)
# conn.close()

# st.dataframe(df)

import streamlit as st
import sqlite3
import pandas as pd

if "doctor_logged_in" not in st.session_state:
    st.error("Please login as doctor first.")
    st.stop()

st.title("📋 Doctor Dashboard")

conn = sqlite3.connect("patients.db")
df = pd.read_sql_query("SELECT * FROM results ORDER BY timestamp DESC", conn)
conn.close()

st.dataframe(df)
