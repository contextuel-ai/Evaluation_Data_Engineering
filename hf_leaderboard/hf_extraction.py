from datasets import load_dataset
import pandas as pd
from datetime import datetime
from pathlib import Path

# Base project directory (llm_demo)
BASE_DIR = Path(__file__).resolve().parent

# Target folder
OUTPUT_DIR = BASE_DIR / "Evaluation_Data_Engineering"

# Ensure it exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset (leaderboard summary table)
dataset = load_dataset("open-llm-leaderboard/contents", split="train")

df = dataset.to_pandas()

print("Columns available:")
print(df.columns)

# Save raw snapshot (for provenance)
timestamp = datetime.utcnow().strftime("%Y%m%d")
output_file = OUTPUT_DIR / f"hf_leaderboard_long_{timestamp}.csv"
df.to_csv(output_file, index=False)

print(f"Saved to {output_file}")