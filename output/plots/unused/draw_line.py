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

# Dictionary for proper model names
model_names = {
    # "mbert": "mBERT",
    # "xlm-roberta": "XLM-RoBERTa",
    "bloom": "BLOOM",
    # "llama2": "LLaMA 2",
    "qwen": "Qwen 2.5",
    "xglm": "XGLM"
}

# Function to compute the bias score
def compute_bias_score(sent_more_score, sent_less_score):
    if (abs(sent_more_score) + abs(sent_less_score) == 0):
        return 1
    else:
        return abs(sent_more_score - sent_less_score) / ((abs(sent_more_score) + abs(sent_less_score)) / 2)

# Initialize a list to store results
plot_data = []

output_dir = "plots"  # Folder to save plots
os.makedirs(output_dir, exist_ok=True)

# Loop through each model and language
for model in model_names.keys():
    dataset_dir = model  # Folder name
    for lang in languages:
        file_path = os.path.join(dataset_dir, f"{lang}.csv")
        
        if os.path.exists(file_path):  # Check if the file exists
            df = pd.read_csv(file_path)
            
            # Compute bias score for each row
            df["bias_score"] = df.apply(lambda row: compute_bias_score(row["sent_more_score"], row["sent_less_score"]), axis=1)
            
            # Compute average bias score per bias type
            bias_avg_scores = df.groupby("bias_type")["bias_score"].mean().to_dict()
            
            # Store the results
            for bias, score in bias_avg_scores.items():
                plot_data.append({"Model": model_names[model], "Language": language_names[lang], "Bias Type": bias, "Bias Score": score})

# Convert results to DataFrame
df_plot = pd.DataFrame(plot_data)

# Plot the line chart
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_plot, x="Bias Type", y="Bias Score", hue="Model", style="Language", markers=True)
plt.title("Bias Score Comparison Across Models, Languages, and Bias Types")
plt.xlabel("Bias Type")
plt.ylabel("Average Bias Score")
plt.xticks(rotation=45)
# plt.legend(title="Model")
plt.grid(True)

# Save the plot
plot_path = os.path.join(output_dir, "bias_score_comparison.png")
plt.savefig(plot_path, dpi=300)
plt.show()