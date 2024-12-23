import pandas as pd
import os

# List of all CSV files to combine
csv_files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

# Combine all CSV files into a single DataFrame
combined_df = pd.DataFrame()
for file in csv_files:
    if os.path.exists(file):
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    else:
        print(f"Warning: {file} not found. Skipping...")

# Save the combined dataset
output_file = "combined_dataset.csv"
combined_df.to_csv(output_file, index=False)
print(f"Combined dataset saved as {output_file}")
