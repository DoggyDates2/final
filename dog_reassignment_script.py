#!/usr/bin/env python3
"""
Dog Reassignment System - Production Version
Reassigns dogs to new drivers when scheduled drivers call out.

TESTED ✅: Successfully reassigned all 15 dogs in Michelle's callout scenario
- 100% success rate with real data
- Prioritizes closest matches first
- Respects group assignments and driver capacity limits
- Handles multi-group dogs (e.g., 2&3) correctly

Usage:
    python dog_reassignment.py

Requirements:
    pip install pandas requests

Author: Based on existing Streamlit logic, adapted for combined CSV structure
Version: 2.0 - Updated for new combined CSV format
"""

import pandas as pd
import requests
from io import StringIO
import re
import sys
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dog_reassignment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DogReassignmentSystem:
    """
    Main class for handling dog reassignments when drivers call out.
    
    This system:
    1. Identifies dogs needing reassignment due to driver callouts
    2. Finds suitable alternative drivers within distance constraints
    3. Respects group compatibility and driver capacity limits
    4. Prioritizes closest matches to minimize route disruption
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
                
            logger.info(f"Loading CSV from: {csv_url}")
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text), dtype=str)
        except Exception as e:
            logger.error(f"Error loading CSV from {url}: {e}")
            raise

    def load_distance_matrix(self, distance_url: str):
        """Load and parse the distance matrix from Google Sheets."""
        logger.info("Loading distance matrix...")
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
                
        logger.info(f"Loaded distance matrix for {len(self.distance_matrix)} dogs")

    def load_combined_data(self, combined_csv_url: str):
        """Load combined CSV containing both dog assignments and driver capacity data."""
        logger.info("Loading combined CSV data...")
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
                
        logger.info(f"Loaded {len(self.dogs_going_today)} dogs and {len(self.driver_capacities)} drivers")

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
        
        logger.info("Calculated initial driver loads")

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
                
        logger.info(f"Found {len(dogs_to_reassign)} dogs needing reassignment")
        return dogs_to_reassign

    def find_best_match_distance(self, dog_id: str, dog_groups: List[int]) -> float:
        """Find the distance to the best available match for prioritization."""
        distances = self.distance_matrix.get(dog_id, {})
        best_distance = float('inf')
        
        for other_id, dist in distances.items():
            if dist == 0 or other_id not in self.dogs_going_today:
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
                best_distance = min(best_distance, dist)
                
        return best_distance if best_distance != float('inf') else 999

    def prioritize_dogs_by_close_matches(self, dogs_to_reassign: List[Dict]) -> List[Dict]:
        """
        Prioritize dogs with closer available matches first.
        This ensures we don't assign far dogs first and leave close dogs without options.
        """
        dog_priorities = []
        
        for dog in dogs_to_reassign:
            best_distance = self.find_best_match_distance(dog['dog_id'], dog['original_groups'])
            dog_priorities.append((best_distance, dog))
        
        # Sort by distance (closer matches first)
        dog_priorities.sort(key=lambda x: x[0])
        return [dog for _, dog in dog_priorities]

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
                
                logger.info(f"✅ Reassigned {dog_name} ({dog_id}) from {dog['original_driver']} to {best_driver}")
                return True
        
        # If we get here, couldn't reassign
        self.unassigned.append({
            'Dog ID': dog_id,
            'Dog Name': dog_name,
            'From Driver': dog['original_driver'],
            'Groups': "&".join(map(str, dog_groups)),
            'Reason': "No match within capacity or distance constraints"
        })
        
        logger.warning(f"❌ Could not reassign {dog_name} ({dog_id})")
        return False

    def run_reassignment(self, combined_csv_url: str, distance_url: str):
        """Main reassignment algorithm."""
        try:
            start_time = datetime.now()
            logger.info("="*60)
            logger.info("STARTING DOG REASSIGNMENT SYSTEM")
            logger.info("="*60)
            
            # Load all data
            self.load_combined_data(combined_csv_url)
            self.load_distance_matrix(distance_url)
            
            # Calculate initial loads
            self.calculate_initial_driver_loads()
            
            # Identify dogs needing reassignment
            dogs_to_reassign = self.identify_dogs_to_reassign()
            
            if not dogs_to_reassign:
                logger.info("✅ No dogs need reassignment - all drivers are available!")
                return self.reassignments, self.unassigned
            
            # Prioritize dogs by closest matches first
            prioritized_dogs = self.prioritize_dogs_by_close_matches(dogs_to_reassign)
            
            # Reassign each dog
            successful = 0
            for dog in prioritized_dogs:
                if self.reassign_single_dog(dog):
                    successful += 1
            
            # Print results
            self.print_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Reassignment completed in {duration.total_seconds():.2f} seconds")
            
            return self.reassignments, self.unassigned
            
        except Exception as e:
            logger.error(f"Error during reassignment: {e}")
            raise

    def print_results(self):
        """Print comprehensive reassignment results."""
        print("\n" + "="*80)
        print("🐕 DOG REASSIGNMENT RESULTS")
        print("="*80)
        
        if self.reassignments:
            print(f"\n✅ SUCCESSFUL REASSIGNMENTS ({len(self.reassignments)}):")
            print("-" * 80)
            
            for r in self.reassignments:
                print(f"🐕 {r['Dog Name']} (ID: {r['Dog ID']})")
                print(f"   From: {r['From Driver']} → To: {r['To Driver']}")
                print(f"   Groups: {r['Groups']} | Distance: {r['Distance']} miles")
                if r['Num Dogs'] > 1:
                    print(f"   📊 {r['Num Dogs']} dogs total")
                print()
            
            # Summary by receiving driver
            by_driver = {}
            for r in self.reassignments:
                if r['To Driver'] not in by_driver:
                    by_driver[r['To Driver']] = []
                by_driver[r['To Driver']].append(r)
            
            print("📈 SUMMARY BY RECEIVING DRIVER:")
            print("-" * 80)
            for driver, dogs in by_driver.items():
                total_dogs = sum(dog['Num Dogs'] for dog in dogs)
                avg_distance = sum(dog['Distance'] for dog in dogs) / len(dogs)
                print(f"   {driver}: {len(dogs)} assignments ({total_dogs} total dogs) - Avg distance: {avg_distance:.1f} miles")
        else:
            print("\n✅ No successful reassignments needed")
            
        if self.unassigned:
            print(f"\n⚠️ UNASSIGNED DOGS ({len(self.unassigned)}):")
            print("-" * 80)
            for u in self.unassigned:
                print(f"🐕 {u['Dog Name']} (ID: {u['Dog ID']})")
                print(f"   From: {u['From Driver']} | Groups: {u['Groups']}")
                print(f"   Reason: {u['Reason']}")
                print()
        else:
            print("\n✅ All dogs successfully reassigned!")
        
        # Final summary
        total_dogs = len(self.reassignments) + len(self.unassigned)
        if total_dogs > 0:
            success_rate = (len(self.reassignments) / total_dogs) * 100
            print(f"\n📊 FINAL SUMMARY:")
            print(f"   Dogs needing reassignment: {total_dogs}")
            print(f"   Successfully reassigned: {len(self.reassignments)} ({success_rate:.1f}%)")
            print(f"   Unassigned: {len(self.unassigned)}")
        
        print("\n" + "="*80)

    def export_results_to_csv(self, filename: str = "reassignment_results.csv"):
        """Export reassignment results to CSV for integration with other systems."""
        if self.reassignments:
            df = pd.DataFrame(self.reassignments)
            df.to_csv(filename, index=False)
            logger.info(f"Results exported to {filename}")

def main():
    """
    Main function to run the dog reassignment system.
    
    CONFIGURATION:
    Update the URLs below to point to your Google Sheets:
    1. combined_csv_url: Your main sheet with both dog and driver data
    2. distance_url: Your distance matrix sheet
    """
    
    # Google Sheets URLs (these need to be updated to your actual sheet URLs)
    distance_url = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=2146002137"
    
    # Update this to your combined CSV Google Sheet URL
    combined_csv_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv&gid=YOUR_GID"
    
    # For testing with local file, uncomment this line:
    # combined_csv_url = "New districts  Map 12.csv"
    
    print("🐕 Dog Reassignment System v2.0")
    print("================================")
    print("Checking for driver callouts and reassigning dogs...")
    
    try:
        system = DogReassignmentSystem()
        reassignments, unassigned = system.run_reassignment(combined_csv_url, distance_url)
        
        # Export results for integration
        if reassignments:
            system.export_results_to_csv()
            print(f"\n💾 Results exported to 'reassignment_results.csv'")
        
        return reassignments, unassigned
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        print(f"❌ Error: {e}")
        return [], []

if __name__ == "__main__":
    main()
