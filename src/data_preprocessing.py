import re
from pathlib import Path

import pandas as pd


# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Define input and output paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "IMDB_Dataset.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_imdb.csv"


def load_data(file_path):
    """
    Load the raw IMDb dataset.
    """
    print("Loading dataset...")

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    return df


def inspect_data(df):
    """
    Display basic information about the dataset.
    """
    print("\n--- Dataset Information ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Sentiment Distribution ---")
    print(df["sentiment"].value_counts())


def clean_text(text):
    """
    Clean a single review.
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags such as <br />
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_data(df):
    """
    Clean and preprocess the dataset.
    """
    print("\nPreprocessing dataset...")

    # Remove missing values
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean review text
    df["cleaned_review"] = df["review"].apply(clean_text)

    # Convert labels into numbers
    df["label"] = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    print("Preprocessing completed.")
    print(f"Final dataset shape: {df.shape}")

    return df


def save_data(df, output_path):
    """
    Save the processed dataset.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nProcessed dataset saved to:")
    print(output_path)


def main():
    df = load_data(RAW_DATA_PATH)

    inspect_data(df)

    cleaned_df = preprocess_data(df)

    save_data(cleaned_df, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()