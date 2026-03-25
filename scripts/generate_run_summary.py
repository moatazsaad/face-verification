import json
import glob
import os

result = []

# Read all JSON files from artifacts/runs directory
for f in glob.glob("artifacts/runs/*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))

os.makedirs("reports", exist_ok=True)

# Write merged output to reports/run_summaries.json
with open("reports/run_summaries.json", "w") as outfile:
    json.dump(result, outfile, indent=4)