import pandas as pd
import os

# Define models and files to process
models = ["mbert", "xlm-roberta", "llama"]
languages = ["en", "zh", "ru", "id", "th"]
bias_types = ["gender", "nationality", "race-color", "religion"]

# Function to compute the bias score
def compute_bias_score(sent_more_score, sent_less_score):
    if (abs(sent_more_score) + abs(sent_less_score) == 0):
        return 1
    else:
        return abs(sent_more_score - sent_less_score) / ((abs(sent_more_score) + abs(sent_less_score)) / 2)

# Initialize a dictionary to store results
all_results = {}

# Loop through each model
for model in models:
    model_dir = model  # Each model has its own folder

    # Loop through each language file
    for lang in languages:
        file_path = os.path.join(model_dir, f"{lang}.csv")
        
        if os.path.exists(file_path):  # Check if the file exists
            df = pd.read_csv(file_path)

            # Compute bias score for each row
            df["bias_score"] = df.apply(lambda row: compute_bias_score(row["sent_more_score"], row["sent_less_score"]), axis=1)

            # Compute average bias score per bias type
            bias_avg_scores = df.groupby("bias_type")["bias_score"].mean().to_dict()
        else:
            # If file is missing, set NaN for all bias types
            bias_avg_scores = {btype: "N/A" for btype in bias_types}

        # Store the results
        all_results[(model, lang)] = bias_avg_scores

# Generate LaTeX table
latex_code = "\\begin{table*}\n"
latex_code += "    \\centering\n"
latex_code += "    \\begin{tabular}{l l " + "c" * len(models) + "}\n"
latex_code += "        \\hline\n"
latex_code += "        Language & Bias Type &" + " & ".join(models) + " \\\\ \\hline\n"

# Populate rows with bias scores per language
for lang in languages:
    first_row = True  # To handle multi-row formatting for the language column
    for bias in bias_types:
        row = [" " if not first_row else lang.capitalize(), bias.capitalize()]
        first_row = False  # After first row, keep language column empty for grouping

        for model in models:
            score = all_results.get((model, lang), {}).get(bias, "N/A")
            if isinstance(score, float):
                row.append(f"{score*100:.3f}")  # Convert to percentage
            else:
                row.append(f"{score}")  # Keep N/A as is

        latex_code += "        " + " & ".join(row) + " \\\\\n"

latex_code += "        \\hline\n"
latex_code += "    \\end{tabular}\n\n"
latex_code += "    \\caption{Bias Score Comparison of Different Models Across Languages}\n"
latex_code += "    \\label{tab:bias-scores}\n"
latex_code += "\\end{table*}\n"

# Print the LaTeX code
print(latex_code)
