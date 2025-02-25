import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the language codes and bias types
languages = ["en", "zh", "ru", "id", "th"]
bias_types = ["gender", "nationality", "race-color", "religion"]

# Dictionary for proper language names
language_names = {
    "en": "English",
    "zh": "Chinese",
    "ru": "Russian",
    "id": "Indonesian",
    "th": "Thai"
}

# Function to compute the bias score
def compute_bias_score(sent_more_score, sent_less_score):
    if (abs(sent_more_score) + abs(sent_less_score) == 0):
        return 1
    else:
        return abs(sent_more_score - sent_less_score) / ((abs(sent_more_score) + abs(sent_less_score)) / 2) *100

# Initialize a list to store the data
all_data = []

# Process only XLM-RoBERTa model
dataset_dir = "xlm-roberta"  # Folder name


# Loop through each language file
for lang in languages:
    file_path = os.path.join(dataset_dir, f"{lang}.csv")
    
    if os.path.exists(file_path):  # Check if the file exists
        df = pd.read_csv(file_path)

        # Compute bias score
        df["bias_score"] = df.apply(lambda row: compute_bias_score(row["sent_more_score"], row["sent_less_score"]), axis=1)

        # Append relevant data for plotting
        for _, row in df.iterrows():
            all_data.append([language_names[lang], row["bias_type"], row["bias_score"]])

# Convert to a DataFrame
df_plot = pd.DataFrame(all_data, columns=["Language", "Bias Type", "Bias Score"])

# Set up the violin plot using sns.catplot
g = sns.catplot(
    data=df_plot, 
    x="Bias Score", 
    y="Bias Type", 
    hue="Language", 
    kind="bar", 
    height= 6,
    aspect = 1.2
    # bw_adjust=0.5, 
    # cut=0, 
    # split=True,
    # aspect=2
)

# Improve the plot aesthetics
g.set_axis_labels("Bias Score", "Bias Type")
g.fig.suptitle("Bias Score Distribution by Bias Type Across Languages in XLM-RoBERTa", fontsize=10)

# g.set(xlim=(0, 0.2))
# # Combine Language and Bias Type for the x-axis
# df_plot["Language_Bias"] = df_plot["Language"] + " (" + df_plot["Bias Type"] + ")"

# # Set up the figure
# plt.figure(figsize=(14, 10))

# # Create the violin plot
# sns.violinplot(
#     data=df_plot,
#     x="Language_Bias",
#     y="Bias Score",
#     hue="Language",  # Ensures each language has the same color
#     palette="Set2",  # Adjust color scheme
#     inner="quart"
# )

# # Improve the plot aesthetics
# plt.xticks(rotation=45, ha="right")
# plt.xlabel("Language (Bias Type)")
# plt.ylabel("Bias Score")
# plt.title("Bias Score Distribution Across Languages and Bias Types")
# plt.legend(title="Language", loc="upper right")

    # Save the plot
output_dir = "xlm-roberta_plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "xlm-roberta_combined.png"))
plt.close()

# Generate radar chart for all languages on the same graph
# plot_radar_chart(all_results)

print("Combined radar chart saved in 'xlm-roberta_plots/' folder.")
