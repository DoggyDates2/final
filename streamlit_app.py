import streamlit as st
from dog_reassignment_final import DogReassignmentSystem
import pandas as pd

st.set_page_config(page_title="Dog Reassignment", layout="wide")

st.title("🐾 Dog Reassignment System")
st.markdown("Paste your CSV export links from Google Sheets:")

combined_url = st.text_input("Combined Sheet CSV Export URL")
distance_url = st.text_input("Distance Matrix CSV Export URL")

if st.button("Run Reassignment"):
    try:
        system = DogReassignmentSystem()
        reassignments, unassigned = system.run_reassignment(combined_url, distance_url)
        st.success("Reassignment complete!")

        if reassignments:
            df = pd.DataFrame(reassignments)
            st.subheader("✅ Reassigned Dogs")
            st.dataframe(df)

        if unassigned:
            df = pd.DataFrame(unassigned)
            st.subheader("⚠️ Unassigned Dogs")
            st.dataframe(df)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
