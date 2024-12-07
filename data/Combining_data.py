
import pandas as pd
from datasets import load_dataset

# Load Wealth Alpaca dataset
wealth_alpaca = load_dataset("gbharti/wealth-alpaca_lora")["train"].to_pandas()

# Load Financial QA dataset
financial_qa = pd.read_csv("data/Financial-QA-10k.csv")

# Standardize columns for compatibility
financial_qa.rename(columns={"question": "instruction", "answer": "output"}, inplace=True)
financial_qa["input"] = ""  # Add an empty 'input' column

# Combine the two datasets
combined_dataset = pd.concat([wealth_alpaca, financial_qa], ignore_index=True)

# Save the combined dataset
combined_dataset.to_csv("data/combined_dataset.csv", index=False)
print("Combined dataset saved to data/combined_dataset.csv")
