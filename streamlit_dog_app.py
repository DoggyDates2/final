# dog_reassignment_final.py

import pandas as pd
import requests
from io import StringIO
import re
from typing import Dict, List, Tuple, Optional

class DogReassignmentSystem:
    def __init__(self):
        self.distance_matrix = {}
        self.dogs_going_today = {}
        self.driver_capacities = {}
        self.driver_callouts = {}
        self.driver_loads = {}
        self.reassignments = []
        self.unassigned = []

    def load_csv_from_url(self, url: str) -> pd.DataFrame:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), dtype=str)

    def load_combined_data(self, combined_csv_url: str):
        combined_df = self.load_csv_from_url(combined_csv_url)

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

        for _, row in combined_df.iterrows():
            driver = str(row.get("R", "")).strip()
            if not driver:
                continue

            def parse_callout(val):
                return str(val).strip().upper() == "X"

            def parse_capacity(val):
                val_str = str(val).strip()
                if val_str.upper() == "X":
                    return 0
                if val_str == "":
                    return 0
                try:
                    return int(float(val_str))
                except:
                    return 0

            g1_val = row.get("U", "")
            g2_val = row.get("V", "")
            g3_val = row.get("W", "")

            self.driver_callouts[driver] = {
                'group1': parse_callout(g1_val),
                'group2': parse_callout(g2_val),
                'group3': parse_callout(g3_val)
            }

            self.driver_capacities[driver] = {
                'group1': parse_capacity(g1_val),
                'group2': parse_capacity(g2_val),
                'group3': parse_capacity(g3_val)
            }

    def load_distance_matrix(self, distance_url: str):
        matrix_df = self.load_csv_from_url(distance_url)
        dog_ids = [str(col).strip() for col in matrix_df.columns[1:]]
        for _, row in matrix_df.iterrows():
            row_id = str(row.iloc[0]).strip()
            self.distance_matrix[row_id] = {}
            for j, col_id in enumerate(dog_ids):
                try:
                    val = float(row.iloc[j + 1])
                except (ValueError, TypeError):
                    val = 0.0
                self.distance_matrix[row_id][col_id] = val

    def parse_group_assignment(self, assignment: str) -> Tuple[Optional[str], List[int]]:
        if ':' not in assignment or 'XX' in assignment:
            return None, []
        driver, groups_str = assignment.split(':', 1)
        groups = sorted(set(int(x) for x in re.findall(r'[123]', groups_str)))
        return driver.strip(), groups

    def calculate_initial_driver_loads(self):
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
        dogs_to_reassign = []
        for dog_id, info in self.dogs_going_today.items():
            driver, groups = self.parse_group_assignment(info['assignment'])
            if not driver or not groups or driver not in self.driver_callouts:
                continue
            callout = self.driver_callouts[driver]
            affected = [g for g in groups if callout.get(f'group{g}', False)]
            if affected:
                dogs_to_reassign.append({
                    'dog_id': dog_id,
                    'original_driver': driver,
                    'original_groups': groups,
                    'affected_groups': affected,
                    'dog_info': info
                })
        return dogs_to_reassign

    def is_adjacent_group(self, group1: int, group2: int) -> bool:
        return (group1 == 1 and group2 == 2) or (group1 == 2 and group2 in [1, 3]) or (group1 == 3 and group2 == 2)

    def can_driver_accommodate(self, driver: str, dog_groups: List[int], num_dogs: int) -> bool:
        for group in dog_groups:
            load = self.driver_loads.get(driver, {}).get(f'group{group}', 0)
            cap = self.driver_capacities.get(driver, {}).get(f'group{group}', 0)
            if load + num_dogs > cap:
                return False
        return True

    def find_candidates_for_dog(self, dog_id: str, dog_groups: List[int], max_distance: float = 3.0) -> List[Tuple[str, float]]:
        candidates = []
        distances = self.distance_matrix.get(dog_id, {})
        for other_id, dist in distances.items():
            if dist == 0 or dist > max_distance or other_id not in self.dogs_going_today:
                continue
            other_info = self.dogs_going_today[other_id]
            other_driver, other_groups = self.parse_group_assignment(other_info['assignment'])
            if not other_driver or not other_groups:
                continue
            if any(self.driver_callouts.get(other_driver, {}).get(f'group{g}', False) for g in dog_groups):
                continue
            matched = []
            for g in dog_groups:
                if g in other_groups or any(self.is_adjacent_group(g, og) for og in other_groups):
                    matched.append(g)
            if set(matched) == set(dog_groups):
                candidates.append((other_driver, dist))
        return candidates

    def reassign_single_dog(self, dog: Dict) -> bool:
        dog_id = dog['dog_id']
        dog_groups = dog['original_groups']
        num_dogs = dog['dog_info']['num_dogs']
        for dist_limit in [0.5, 1.0, 1.5, 2.0, 3.0]:
            candidates = self.find_candidates_for_dog(dog_id, dog_groups, dist_limit)
            viable = [(drv, d) for drv, d in candidates if self.can_driver_accommodate(drv, dog_groups, num_dogs)]
            if viable:
                viable.sort(key=lambda x: x[1])
                chosen = viable[0][0]
                for g in dog_groups:
                    self.driver_loads.setdefault(chosen, {'group1': 0, 'group2': 0, 'group3': 0})
                    self.driver_loads[chosen][f'group{g}'] += num_dogs
                self.reassignments.append({
                    'Dog ID': dog_id,
                    'Dog Name': dog['dog_info']['dog_name'],
                    'From Driver': dog['original_driver'],
                    'To Driver': chosen,
                    'Groups': "&".join(map(str, dog_groups)),
                    'Distance': round(viable[0][1], 2),
                    'Num Dogs': num_dogs
                })
                return True
        self.unassigned.append({
            'Dog ID': dog_id,
            'Dog Name': dog['dog_info']['dog_name'],
            'From Driver': dog['original_driver'],
            'Groups': "&".join(map(str, dog_groups)),
            'Reason': "No viable match within distance/capacity"
        })
        return False

    def run_reassignment(self, combined_csv_url: str, distance_url: str):
        self.load_combined_data(combined_csv_url)
        self.load_distance_matrix(distance_url)
        self.calculate_initial_driver_loads()
        dogs_to_reassign = self.identify_dogs_to_reassign()
        for dog in dogs_to_reassign:
            self.reassign_single_dog(dog)
        return self.reassignments, self.unassigned
