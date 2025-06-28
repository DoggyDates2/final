import streamlit as st
import pandas as pd
from dog_reassignment import DogReassignmentSystem
import logging

# Configure page
st.set_page_config(
    page_title="Dog Reassignment System", 
    layout="wide",
    page_icon="🐕"
)

st.title("🐕 Dog Reassignment System")
st.markdown("Automatically reassign dogs when drivers call out")

# Sidebar for inputs
st.sidebar.header("🔗 Configuration")

# URL inputs
combined_csv_url = st.sidebar.text_input(
    "Combined CSV URL (Dog + Driver Data)", 
    value="https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv&gid=YOUR_GID",
    help="Your Google Sheet with both dog assignments and driver capacity data"
)

distance_url = st.sidebar.text_input(
    "Distance Matrix URL", 
    value="https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=2146002137",
    help="Your distance matrix Google Sheet"
)

# Run button
if st.sidebar.button("🔄 Run Reassignment", type="primary"):
    
    if not combined_csv_url or "YOUR_SHEET_ID" in combined_csv_url:
        st.error("⚠️ Please update the Combined CSV URL with your actual Google Sheet URL")
        st.stop()
    
    try:
        with st.spinner("Loading data and running reassignment..."):
            # Create system and run reassignment
            system = DogReassignmentSystem()
            reassignments, unassigned = system.run_reassignment(combined_csv_url, distance_url)
        
        # Display results
        if reassignments or unassigned:
            st.success(f"✅ Reassignment complete!")
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Successful Reassignments")
                if reassignments:
                    df_success = pd.DataFrame(reassignments)
                    st.dataframe(df_success, use_container_width=True)
                    
                    # Summary by driver
                    st.subheader("📈 Summary by Driver")
                    summary = df_success.groupby('To Driver').agg({
                        'Dog ID': 'count',
                        'Num Dogs': 'sum',
                        'Distance': 'mean'
                    }).round(2)
                    summary.columns = ['Assignments', 'Total Dogs', 'Avg Distance']
                    st.dataframe(summary, use_container_width=True)
                    
                    # Download button
                    csv = df_success.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="reassignment_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No successful reassignments needed")
            
            with col2:
                st.subheader("⚠️ Unassigned Dogs")
                if unassigned:
                    df_unassigned = pd.DataFrame(unassigned)
                    st.dataframe(df_unassigned, use_container_width=True)
                else:
                    st.success("All dogs successfully reassigned!")
            
            # Overall statistics
            total_dogs = len(reassignments) + len(unassigned)
            if total_dogs > 0:
                success_rate = (len(reassignments) / total_dogs) * 100
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Dogs Reassigned", len(reassignments))
                with col2:
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col3:
                    st.metric("Unassigned", len(unassigned))
        
        else:
            st.success("🎉 No driver callouts detected - all drivers are available!")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# Instructions section
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Setup Instructions:
    1. **Update URLs**: Replace the placeholder URLs in the sidebar with your actual Google Sheets URLs
    2. **Sheet Format**: Ensure your CSV has the required columns:
       - Dog data: `Dog ID`, `Today`, `Number of dogs`, `Dog Name`, `Address`
       - Driver data: `Driver`, `Group 1`, `Group 2`, `Group 3` (use 'X' for callouts)
    3. **Click Run**: Press the "Run Reassignment" button to process
    
    ### Features:
    - ✅ Prioritizes closest matches first
    - ✅ Handles multi-group dogs (1&2, 2&3, etc.)
    - ✅ Respects driver capacity limits  
    - ✅ Exports results to CSV
    
    ### Callout Format:
    Mark driver callouts by putting **'X'** in the Group columns:
    - Group 1: X (called out for morning group)
    - Group 2: X (called out for late morning group)  
    - Group 3: X (called out for afternoon group)
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for efficient dog logistics")
