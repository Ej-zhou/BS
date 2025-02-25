import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

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
        return abs(sent_more_score - sent_less_score) / ((abs(sent_more_score) + abs(sent_less_score)) / 2) * 100

# Initialize a dictionary to store results
all_results = {}

# Process only XLM-RoBERTa model
dataset_dir = "xglm"  # Folder name

# Loop through each language file
for lang in languages:
    file_path = os.path.join(dataset_dir, f"{lang}.csv")
    
    if os.path.exists(file_path):  # Check if the file exists
        df = pd.read_csv(file_path)
        
        # Compute bias score for each row
        df["bias_score"] = df.apply(lambda row: compute_bias_score(row["sent_more_score"], row["sent_less_score"]), axis=1)
        
        # Compute average bias score per bias type
        bias_avg_scores = df.groupby("bias_type")["bias_score"].mean().to_dict()
        
        # Store the results
        all_results[lang] = bias_avg_scores
    
# Radar chart function
def plot_radar_chart(results):
    labels = bias_types
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles.append(angles[0])
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    
    colors = ['b', 'r', 'g', 'purple', 'orange']
    
    for i, (lang, scores) in enumerate(results.items()):
        values = [scores.get(bt, 0) for bt in labels]
        values.append(values[0])  # Close the circle
        
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.3, label=language_names[lang])
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Bias Scores for XLM-RoBERTa Across Languages")
    ax.legend(loc="upper right")
    
    ax.set_ylim(0.5, 4)

    # Save the plot
    output_dir = "xlm-roberta_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "xlm-roberta_combined.png"))
    plt.close()

# Generate radar chart for all languages on the same graph
plot_radar_chart(all_results)

print("Combined radar chart saved in 'xlm-roberta_plots/' folder.")
