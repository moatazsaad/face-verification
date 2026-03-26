import json
import glob
import os


result = []

# Read all JSON files from artifacts/runs directory
for f in glob.glob("artifacts/runs/*.json"):
    with open(f, "r") as infile:
        data = json.load(infile)
        result.append(data)

# Sort the merged list by the "timestamp" key
result_sorted = sorted(result, key=lambda x: x["timestamp"])

# Ensure the reports directory exists
os.makedirs("reports", exist_ok=True)

# Write merged and sorted output to reports/run_summaries.json
with open("reports/run_summaries.json", "w") as outfile:
    json.dump(result_sorted, outfile, indent=4)

print("Merged and sorted JSON written to reports/run_summaries.json")