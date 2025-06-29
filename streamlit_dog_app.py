# Extract unique drivers from the Today column (who actually have assignments today)
unique_drivers = set()
for today_val in combined_df['Today'].dropna():
    if ':' in today_val:
        driver = today_val.split(":")[0].strip()
        unique_drivers.add(driver)

for driver in unique_drivers:
    # Search the first row that matches the driver in the Today column
    matching_row = combined_df[combined_df['Today'].str.startswith(driver + ':', na=False)].head(1)
    if matching_row.empty:
        continue

    row = matching_row.iloc[0]

    def parse_group(val):
        return str(val).strip().upper() if pd.notna(val) else ""

    g1 = parse_group(row.get("Group 1", ""))
    g2 = parse_group(row.get("Group 2", ""))
    g3 = parse_group(row.get("Group 3", ""))

    self.driver_callouts[driver] = {
        'group1': g1 == "X",
        'group2': g2 == "X",
        'group3': g3 == "X"
    }

    def parse_capacity(val):
        if val == "X":
            return 9  # Still used for logic
        return 9 if val == "" else int(float(val))

    self.driver_capacities[driver] = {
        'group1': parse_capacity(g1),
        'group2': parse_capacity(g2),
        'group3': parse_capacity(g3)
    }
