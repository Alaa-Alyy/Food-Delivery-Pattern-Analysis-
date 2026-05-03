import pandas as pd
import re

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("ULTRA_FINAL_REVIEWS.csv")

# -----------------------------
# Step 1: Lowercase
# -----------------------------
df["clean"] = df["Review"].str.lower()

# -----------------------------
# Step 2: Fix common typos (partial)
# -----------------------------
typo_map = {
    "teh": "the",
    "recieve": "receive",
    "definately": "definitely"
}

for wrong, correct in typo_map.items():
    df["clean"] = df["clean"].str.replace(wrong, correct)

# -----------------------------
# Step 3: Emoji Handling
# -----------------------------
emoji_map = {
    "🔥": " fire ",
    "😡": " angry ",
    "😍": " love ",
    "🤢": " disgust ",
    "👍": " good ",
    "👎": " bad ",
    "😐": " neutral "
}

for emoji, word in emoji_map.items():
    df["clean"] = df["clean"].str.replace(emoji, word)

# -----------------------------
# Step 4: Remove extra spaces
# -----------------------------
df["clean"] = df["clean"].apply(lambda x: re.sub(r"\s+", " ", x).strip())

# -----------------------------
# Step 5: Keep punctuation but clean noise
# -----------------------------
df["clean"] = df["clean"].apply(lambda x: re.sub(r"[^\w\s.,!?]", "", x))

# -----------------------------
# Step 6: Optional (Normalize repeated letters)
# ex: gooooood -> good
# -----------------------------
def normalize_text(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

df["clean"] = df["clean"].apply(normalize_text)

# -----------------------------
# Save Cleaned Data
# -----------------------------
df.to_csv("CLEAN_REVIEWS.csv", index=False)

print(df[["Review", "clean"]].head())