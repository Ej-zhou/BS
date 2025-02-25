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
            
bias_matrix = df.pivot_table(index="bias_type", columns="language", values="bias_score", aggfunc="mean")
plt.figure(figsize=(8, 6))
sns.heatmap(bias_matrix, annot=True, cmap="coolwarm", center=1)
plt.title(f"Bias Score Heatmap for {model_names[model]}")
plt.xlabel("Language")
plt.ylabel("Bias Type")
plt.savefig(os.path.join(output_dir, f"{model}_bias_heatmap.png"))
plt.show()