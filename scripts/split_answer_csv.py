import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = r"C:\Users\Fatin\OneDrive\Desktop\CAT405\FYP\data\answer.csv"
OUTPUT_DIR = r"C:\Users\Fatin\OneDrive\Desktop\CAT405\FYP\data\dg"
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

required_cols = {"input_text", "target_text"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["input_text", "target_text"]).copy()
df["input_text"] = df["input_text"].astype(str).str.strip()
df["target_text"] = df["target_text"].astype(str).str.strip()
df = df[(df["input_text"] != "") & (df["target_text"] != "")]

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, shuffle=True)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("Done.")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")