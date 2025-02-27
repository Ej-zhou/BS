import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define language codes and bias types
LANGUAGES = ["en", "zh", "ru", "id", "th"]
BIAS_TYPES = ["gender", "nationality", "race-color", "religion"]

# Dictionary for proper language names
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ru": "Russian",
    "id": "Indonesian",
    "th": "Thai"
}

BIAS_NAMES = {
    "gender": "Gender",
    "nationality": "Nationality",
    "race-color" :"Race-Color",
    "religion" : "Religion"

}

# Function to compute bias score
def compute_bias_score(sent_more_score, sent_less_score):
    denominator = abs(sent_more_score) + abs(sent_less_score)
    return 1 if denominator == 0 else abs(sent_more_score - sent_less_score) / (denominator / 2) * 100

# Load and process dataset
def load_data(dataset_dir="mbert"):
    """Loads and processes bias scores from CSV files."""
    all_data = []

    for lang in LANGUAGES:
        file_path = os.path.join(dataset_dir, f"{lang}.csv")
        if os.path.exists(file_path):  # Ensure file exists
            df = pd.read_csv(file_path)

            # Compute bias score and store results
            df["bias_score"] = df.apply(lambda row: compute_bias_score(row["sent_more_score"], row["sent_less_score"]), axis=1)
            df_filtered = df[df["bias_score"] > 0]  # Filter out zero or negative scores

            all_data.extend([[LANGUAGE_NAMES[lang], row["bias_type"], row["bias_score"]] for _, row in df_filtered.iterrows()])

    return pd.DataFrame(all_data, columns=["Language", "Bias Type", "Bias Score"])

# Function to plot bias distribution
def plot_bias_distribution(df_plot, figsize=(12, 6), output_dir="xlm-roberta_plots"):
    """Generates and saves a violin plot for bias score distribution."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    df_plot["Bias Type"] = df_plot["Bias Type"].map(BIAS_NAMES)  # Convert to proper names
    g = sns.catplot(
        data=df_plot,
        y="Bias Score",
        kind="violin",
        col="Bias Type",
        hue="Language",
        bw_adjust=0.5,
        cut=0,
        fill=True,
        native_scale=False,
        log_scale=True,
        height=figsize[1] / 2,  # Adjusting height dynamically
        aspect=figsize[0] / figsize[1] / len(BIAS_TYPES),  # Ensuring proper aspect ratio
        # legend=False
    )

    g.set_titles(col_template="{col_name}", size=14) 
    g.set_axis_labels("","log-scaled Bias Score", size=14)
    # plt.tight_layout() 

    for ax in g.axes.flat:
        ax.set_ylim(10**-1.5, 10**2)  # Example: Adjust the x-axis range (log scale)
        ax.set_xticks([]) 
    # g.add_legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend outside

    # Save plot
    output_path = os.path.join(output_dir, "VIOLIN.png")
    plt.savefig(output_path,bbox_inches="tight", dpi=300,)
    plt.close()
    print(f"Chart saved in '{output_path}'")

# Main execution
if __name__ == "__main__":
    df_plot = load_data("bloom")
    plot_bias_distribution(df_plot, figsize=(25, 7))  # Adjust figsize as needed