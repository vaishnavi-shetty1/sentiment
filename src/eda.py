from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_imdb.csv"
OUTPUT_DIR = BASE_DIR / "artifacts" / "eda"


def load_data():
    print("Loading processed dataset...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=["cleaned_review", "label"]
    )

    print(f"Dataset shape: {df.shape}")

    return df


def analyze_data(df):
    print("\n--- Dataset Information ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Label Distribution ---")
    print(df["label"].value_counts())

    print("\n--- Label Percentage ---")
    print(
        df["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


def add_review_length(df):
    df["review_length"] = (
        df["cleaned_review"]
        .fillna("")
        .str.split()
        .str.len()
    )

    print("\n--- Review Length Statistics ---")
    print(df["review_length"].describe())

    return df


def plot_sentiment_distribution(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))

    plt.bar(
        ["Negative", "Positive"],
        counts.values
    )

    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")

    output_path = OUTPUT_DIR / "sentiment_distribution.png"

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


def plot_review_length(df):
    plt.figure(figsize=(8, 5))

    plt.hist(
        df["review_length"],
        bins=50
    )

    plt.title("Distribution of Review Length")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    output_path = OUTPUT_DIR / "review_length_distribution.png"

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    df = load_data()

    analyze_data(df)

    df = add_review_length(df)

    plot_sentiment_distribution(df)

    plot_review_length(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()