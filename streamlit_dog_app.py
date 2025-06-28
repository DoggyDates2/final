import streamlit as st
import pandas as pd
import requests
from io import StringIO
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Dog Reassignment System", 
    layout="wide",
    page_icon="🐕"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DogReassignmentSystem:
    """
    Dog Reassignment System - All functions in one file to avoid import issues
    """
    
    def __init__(self):
        self.distance_matrix = {}
        self.dogs_going_today = {}
        self.driver_capacities = {}
        self.driver_callouts = {}
        self.driver_loads = {}
        self.reassignments = []
        self.unassigned = []
        
    def load_csv_from_url(self, url: str) -> pd.DataFrame:
        """Convert Google Sheets URL to CSV export format and load data."""
        try:
            if "docs.google.com" in url and "edit" in url:
                sheet_id = url.split("/d/")[1].split("/")[0]
                gid = url.split("gid=")[-1].split("#")[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            else:
                csv_url = url
                
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text), dtype=str)
        except Exception as e:
            logger.error(f"Error loading CSV from {url}: {e}")
            raise

    def load_distance_matrix(self, distance_url: str):
        """Load and parse the distance matrix from Google Sheets."""
        matrix_df = self.load_csv_from_url(distance_url)
        
        # First column contains dog IDs, remaining columns are distances
        dog_ids = [str(col).strip() for col in matrix_df.columns[1:]]
        
        for i, row in matrix_df.iterrows():
            row_id = str(row.iloc[0]).strip()
            self.distance_matrix[row_id] = {}
            
            for j, col_id in enumerate(dog_ids):
                try:
                    val = float(row.iloc[j + 1])
                except (ValueError, TypeError):
                    val = 0.0  # 0 means not viable match
                self.distance_matrix[row_id][col_id] = val

    def load_combined_data(self, combined_csv_url: str):
        """Load combined CSV containing both dog assignments and driver capacity data."""
        combined_df = self.load_csv_from_url(combined_csv_url)
        
        # Extract dogs going today from the CSV
        for _, row in combined_df.iterrows():
            dog_id = str(row.get("Dog ID", "")).strip()
            assignment = str(row.get("Today", "")).strip()
            
            try:
                num_dogs = int(float(row.get("Number of dogs", "1")))
            except (ValueError, TypeError):
                num_dogs = 1
                
            if dog_id and assignment and ":" in assignment and "XX" not in assignment:
                self.dogs_going_today[dog_id] = {
                    'assignment': assignment,
                    'num_dogs': num_dogs,
                    'address': str(row.get("Address", "")),
                    'dog_name': str(row.get("Dog Name", ""))
                }
        
        # Extract driver capacity and callout data from the same CSV
        drivers_processed = set()
        for _, row in combined_df.iterrows():
            driver = str(row.get("Driver", "")).strip()
            if not driver or driver in drivers_processed:
                continue
            drivers_processed.add(driver)
                
            def parse_group(val):
                return str(val).strip().upper() if pd.notna(val) else ""
            
            g1 = parse_group(row.get("Group 1", ""))
            g2 = parse_group(row.get("Group 2", ""))
            g3 = parse_group(row.get("Group 3", ""))
            
            # Track callouts (X means driver called out for that group)
            self.driver_callouts[driver] = {
                'group1': g1 == "X",
                'group2': g2 == "X", 
                'group3': g3 == "X"
            }
            
            # Parse capacities (X or empty means default capacity of 9)
            def parse_capacity(val):
                if val == "X":
                    return 9  # Called out drivers still have capacity for logic purposes
                return 9 if val == "" else int(float(val))
            
            self.driver_capacities[driver] = {
                'group1': parse_capacity(g1),
                'group2': parse_capacity(g2),
                'group3': parse_capacity(g3)
            }

    def parse_group_assignment(self, assignment: str) -> Tuple[Optional[str], List[int]]:
        """Parse assignment like 'Andy:1&2' into driver name and list of groups."""
        if ':' not in assignment or 'XX' in assignment:
            return None, []
            
        driver, groups_str = assignment.split(':', 1)
        groups = sorted(set(int(x) for x in re.findall(r'[123]', groups_str)))
        return driver.strip(), groups

    def is_adjacent_group(self, group1: int, group2: int) -> bool:
        """Check if two groups are adjacent (can be substituted with half importance)."""
        return (group1 == 1 and group2 == 2) or \
               (group1 == 2 and group2 in [1, 3]) or \
               (group1 == 3 and group2 == 2)

    def calculate_initial_driver_loads(self):
        """Calculate current driver loads before any reassignments."""
        self.driver_loads = {}
        
        for dog_id, info in self.dogs_going_today.items():
            driver, groups = self.parse_group_assignment(info['assignment'])
            if not driver or not groups:
                continue
                
            if driver not in self.driver_loads:
                self.driver_loads[driver] = {'group1': 0, 'group2': 0, 'group3': 0}
                
            for group in groups:
                self.driver_loads[driver][f'group{group}'] += info['num_dogs']

    def identify_dogs_to_reassign(self) -> List[Dict]:
        """Identify dogs that need reassignment due to driver callouts."""
        dogs_to_reassign = []
        
        for dog_id, info in self.dogs_going_today.items():
            driver, groups = self.parse_group_assignment(info['assignment'])
            if not driver or not groups or driver not in self.driver_callouts:
                continue
                
            callout = self.driver_callouts[driver]
            affected_groups = []
            
            for group in groups:
                if (group == 1 and callout['group1']) or \
                   (group == 2 and callout['group2']) or \
                   (group == 3 and callout['group3']):
                    affected_groups.append(group)
                    
            if affected_groups:
                dogs_to_reassign.append({
                    'dog_id': dog_id,
                    'original_driver': driver,
                    'original_groups': groups,
                    'affected_groups': affected_groups,
                    'dog_info': info
                })
                
        return dogs_to_reassign

    def can_driver_accommodate(self, driver: str, dog_groups: List[int], num_dogs: int) -> bool:
        """Check if driver can accommodate additional dogs without exceeding capacity."""
        for group in dog_groups:
            group_key = f'group{group}'
            current_load = self.driver_loads.get(driver, {}).get(group_key, 0)
            max_capacity = self.driver_capacities.get(driver, {}).get(group_key, 9)
            
            if current_load + num_dogs > max_capacity:
                return False
                
        return True

    def find_candidates_for_dog(self, dog_id: str, dog_groups: List[int], 
                               max_distance: float = 3.0) -> List[Tuple[str, float]]:
        """Find candidate drivers for a dog within distance and group constraints."""
        candidates = []
        distances = self.distance_matrix.get(dog_id, {})
        
        for other_id, dist in distances.items():
            if dist == 0 or dist > max_distance or other_id not in self.dogs_going_today:
                continue
                
            other_info = self.dogs_going_today[other_id]
            other_driver, other_groups = self.parse_group_assignment(other_info['assignment'])
            
            if not other_driver or not other_groups:
                continue
                
            # Skip if other driver also called out for any of dog's groups
            if any(self.driver_callouts.get(other_driver, {}).get(f'group{g}', False) 
                   for g in dog_groups):
                continue
                
            # Check group compatibility
            matched_groups = []
            for dog_group in dog_groups:
                if dog_group in other_groups:
                    matched_groups.append(dog_group)
                elif any(self.is_adjacent_group(dog_group, og) for og in other_groups):
                    matched_groups.append(dog_group)
                    
            # Only consider if all dog's groups can be matched
            if set(matched_groups) == set(dog_groups):
                candidates.append((other_driver, dist))
                
        return candidates

    def get_closest_distance_to_driver(self, driver: str, dog_id: str) -> float:
        """Get distance from dog to closest existing dog for this driver."""
        min_distance = float('inf')
        
        for other_id, info in self.dogs_going_today.items():
            other_driver, _ = self.parse_group_assignment(info['assignment'])
            if other_driver == driver:
                distance = self.distance_matrix.get(dog_id, {}).get(other_id, float('inf'))
                min_distance = min(min_distance, distance)
                
        return min_distance if min_distance != float('inf') else 0

    def reassign_single_dog(self, dog: Dict) -> bool:
        """Attempt to reassign a single dog to the best available driver."""
        dog_id = dog['dog_id']
        dog_name = dog['dog_info']['dog_name']
        dog_groups = dog['original_groups']
        num_dogs = dog['dog_info']['num_dogs']
        
        # Start with tight constraints, expand if needed
        distance_thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        for max_dist in distance_thresholds:
            candidates = self.find_candidates_for_dog(dog_id, dog_groups, max_dist)
            
            if not candidates:
                continue
                
            # Filter candidates that can accommodate the dog
            viable_candidates = []
            
            for driver, dist in candidates:
                if not self.can_driver_accommodate(driver, dog_groups, num_dogs):
                    continue
                    
                # Calculate selection criteria
                current_load = sum(self.driver_loads.get(driver, {}).get(f'group{g}', 0) 
                                 for g in dog_groups)
                closest_dist = self.get_closest_distance_to_driver(driver, dog_id)
                
                viable_candidates.append((driver, current_load, closest_dist))
            
            if viable_candidates:
                # Sort by: 1) fewest dogs currently assigned, 2) closest distance to existing dogs
                viable_candidates.sort(key=lambda x: (x[1], x[2]))
                best_driver = viable_candidates[0][0]
                best_distance = self.get_closest_distance_to_driver(best_driver, dog_id)
                
                # Update driver loads
                for group in dog_groups:
                    if best_driver not in self.driver_loads:
                        self.driver_loads[best_driver] = {'group1': 0, 'group2': 0, 'group3': 0}
                    self.driver_loads[best_driver][f'group{group}'] += num_dogs
                
                self.reassignments.append({
                    'Dog ID': dog_id,
                    'Dog Name': dog_name,
                    'From Driver': dog['original_driver'],
                    'To Driver': best_driver,
                    'Groups': "&".join(map(str, dog_groups)),
                    'Distance': round(best_distance, 3),
                    'Num Dogs': num_dogs
                })
                
                return True
        
        # If we get here, couldn't reassign
        self.unassigned.append({
            'Dog ID': dog_id,
            'Dog Name': dog_name,
            'From Driver': dog['original_driver'],
            'Groups': "&".join(map(str, dog_groups)),
            'Reason': "No match within capacity or distance constraints"
        })
        
        return False

    def prioritize_dogs_by_close_matches(self, dogs_to_reassign: List[Dict]) -> List[Dict]:
        """Prioritize dogs with closer available matches first."""
        def get_best_distance(dog):
            dog_id = dog['dog_id']
            dog_groups = dog['original_groups']
            distances = self.distance_matrix.get(dog_id, {})
            best_distance = float('inf')
            
            for other_id, dist in distances.items():
                if dist == 0 or other_id not in self.dogs_going_today:
                    continue
                    
                other_info = self.dogs_going_today[other_id]
                other_driver, other_groups = self.parse_group_assignment(other_info['assignment'])
                
                if not other_driver or not other_groups:
                    continue
                    
                # Skip if other driver also called out
                if any(self.driver_callouts.get(other_driver, {}).get(f'group{g}', False) 
                       for g in dog_groups):
                    continue
                    
                # Check group compatibility
                matched_groups = []
                for dog_group in dog_groups:
                    if dog_group in other_groups:
                        matched_groups.append(dog_group)
                    elif any(self.is_adjacent_group(dog_group, og) for og in other_groups):
                        matched_groups.append(dog_group)
                        
                if set(matched_groups) == set(dog_groups):
                    best_distance = min(best_distance, dist)
                    
            return best_distance if best_distance != float('inf') else 999
        
        # Sort by distance (closer matches first)
        dogs_to_reassign.sort(key=get_best_distance)
        return dogs_to_reassign

    def run_reassignment(self, combined_csv_url: str, distance_url: str):
        """Main reassignment algorithm."""
        # Load all data
        self.load_combined_data(combined_csv_url)
        self.load_distance_matrix(distance_url)
        
        # Calculate initial loads
        self.calculate_initial_driver_loads()
        
        # Identify dogs needing reassignment
        dogs_to_reassign = self.identify_dogs_to_reassign()
        
        if not dogs_to_reassign:
            return self.reassignments, self.unassigned
        
        # Prioritize dogs by closest matches first
        prioritized_dogs = self.prioritize_dogs_by_close_matches(dogs_to_reassign)
        
        # Reassign each dog
        for dog in prioritized_dogs:
            self.reassign_single_dog(dog)
        
        return self.reassignments, self.unassigned

# Helper function to convert Google Sheets URL to CSV
def convert_to_csv_url(sheets_url):
    """Convert Google Sheets URL to CSV export format"""
    if not sheets_url or "docs.google.com" not in sheets_url:
        return sheets_url
    
    try:
        if "/edit" in sheets_url:
            # Extract sheet ID and gid
            sheet_id = sheets_url.split("/d/")[1].split("/")[0]
            if "#gid=" in sheets_url:
                gid = sheets_url.split("#gid=")[1]
            elif "gid=" in sheets_url:
                gid = sheets_url.split("gid=")[1].split("&")[0]
            else:
                gid = "0"  # Default to first sheet
            
            return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    except:
        pass
    
    return sheets_url

# STREAMLIT APP STARTS HERE
st.title("🐕 Dog Reassignment System")
st.markdown("Automatically reassign dogs when drivers call out")

# Sidebar for inputs
st.sidebar.header("🔗 Configuration")
st.sidebar.subheader("📊 Data Sources")

# Combined CSV URL
combined_sheets_url = st.sidebar.text_input(
    "Combined Sheet URL (Regular Google Sheets Link)", 
    value="",
    help="Paste your regular Google Sheets URL here - we'll convert it to CSV format automatically"
)

if combined_sheets_url:
    combined_csv_url = convert_to_csv_url(combined_sheets_url)
    st.sidebar.success("✅ Converted to CSV format:")
    st.sidebar.code(combined_csv_url, language=None)
else:
    combined_csv_url = ""

# Distance Matrix URL  
distance_sheets_url = st.sidebar.text_input(
    "Distance Matrix URL (Regular Google Sheets Link)", 
    value="https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/edit#gid=2146002137",
    help="Your distance matrix Google Sheet URL"
)

distance_url = convert_to_csv_url(distance_sheets_url)
if distance_sheets_url:
    st.sidebar.success("✅ Distance matrix CSV format:")
    st.sidebar.code(distance_url, language=None)

# Save URLs button
if st.sidebar.button("💾 Save URLs"):
    if combined_csv_url and distance_url:
        # Create a config file content
        config_content = f"""# Dog Reassignment System URLs
# Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

COMBINED_CSV_URL={combined_csv_url}
DISTANCE_MATRIX_URL={distance_url}

# Use these URLs in your script or save this as config.env
"""
        st.sidebar.download_button(
            label="📥 Download URL Config",
            data=config_content,
            file_name="dog_reassignment_urls.txt",
            mime="text/plain"
        )
        st.sidebar.success("✅ URLs ready for download!")
    else:
        st.sidebar.error("Please enter both URLs first")

# Run button
if st.sidebar.button("🔄 Run Reassignment", type="primary"):
    
    if not combined_csv_url or not combined_sheets_url:
        st.error("⚠️ Please enter your Combined Sheet URL first")
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
        with st.expander("🔍 Error Details"):
            st.exception(e)

# Instructions section
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Setup Instructions:
    1. **Paste URLs**: Copy your Google Sheets URLs from your browser and paste them above
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
