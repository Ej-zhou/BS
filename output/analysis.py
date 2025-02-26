import pandas as pd
import os

# Define models and files to process
models = ["mbert", "xlm-roberta", "xglm" , "bloom",  "qwen", "llama2","llama3"]
languages = ["en", "zh", "ru", "id", "th"]
bias_types = ["gender", "nationality", "race-color", "religion"]

# Dictionary for proper model names
model_names = {
    "mbert": "mBERT",
    "xlm-roberta": "XLM-RoBERTa",
    "bloom": "BLOOM",
    "llama2": "LLaMA 2",
    "qwen": "Qwen 2.5",
    "xglm": "XGLM",
    "llama3": "LLaMA 3"
}

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

            # Compute the overall weighted average bias score (mean of all instances)
            total_instances = len(df)
            overall_avg = df["bias_score"].sum() / total_instances if total_instances > 0 else "N/A"

            # Add overall average to the dictionary
            bias_avg_scores["average"] = overall_avg

        else:
            # If file is missing, set NaN for all bias types
            bias_avg_scores = {btype: "N/A" for btype in bias_types}
            bias_avg_scores["Average"] = "N/A"

        # Store the results
        all_results[(model, lang)] = bias_avg_scores

#----------------------without average----------------------
#         # Generate LaTeX table
# latex_code = "\\begin{table*}\n"
# latex_code += "    \\centering\n"
# latex_code += "    \\begin{tabular}{l l " + "c" * len(models) + "}\n"
# latex_code += "        \\hline\n"
# latex_code += "        \\multirow{2}{*}{Language} & \\multirow{2}{*}{Bias Type} & \\multicolumn{" + str(len(models)) + "}{c}{Bias Score per Model} \\\\ \n"
# latex_code += "        & & " + " & ".join(model_names[model] for model in models) + " \\\\ \\hline \\hline \n"

# Populate rows with bias scores per language
# for lang in languages:
#     first_row = True  # To handle multi-row formatting for the language column
#     for i, bias in enumerate(bias_types):  # Include "average" row
#         row = []
#         if i == 0:
#             row.append(f"\\multirow{{{len(bias_types) + 1}}}{{*}}{{{language_names[lang].capitalize()}}}")  # Multirow for language name
#         else:
#             row.append(" ")  # Empty column for bias type rows

#         # if bias == "average":
#         #     row.append("\\textbf{Average}")  # Bold the "Average" label
#         # else:
#         row.append(bias.capitalize())  # Bias Type

#         for model in models:
#             score = all_results.get((model, lang), {}).get(bias, "N/A")
#             if isinstance(score, float):
#                 # if bias == "average":
#                 #     row.append(f"\\textbf{{{score*100:.3f}}}")  # Bold the average values
#                 # else:
#                     row.append(f"{score*100:.3f}")  # Convert to percentage
#             else:
#                 # if bias == "average":
#                 #     row.append("\\textbf{N/A}")  # Keep bold N/A for average row
#                 # else:
#                 row.append(f"{score}")  # Keep N/A as is

#         latex_code += "        " + " & ".join(row) + " \\\\\n"
#     latex_code += "        \\hline\n"  # Add hline at the end of each language block

# latex_code += "  \\hline  \\end{tabular}\n\n"
# latex_code += "    \\caption{Bias Score Comparison of Different Models Across Languages}\n"
# latex_code += "    \\label{tab:bias-scores}\n"
# latex_code += "\\end{table*}\n"
#----------------------without average----------------------


#----------------------with average----------------------
        # Generate LaTeX table
latex_code = "\\begin{table*}\n"
latex_code += "    \\centering\n"
latex_code += "    \\begin{tabular}{l l " + "c" * len(models) + "}\n"
latex_code += "        \\hline\n"
latex_code += "        \\multirow{2}{*}{Language} & \\multirow{2}{*}{Bias Type} & \\multicolumn{" + str(len(models)) + "}{c}{Bias Score per Model} \\\\ \n"
latex_code += "        & & " + " & ".join(model_names[model] for model in models) + " \\\\ \\hline \\hline \n"

# Populate rows with bias scores per language
for lang in languages:
    first_row = True  # To handle multi-row formatting for the language column
    for i, bias in enumerate(bias_types + ["average"]):  # Include "average" row
        row = []
        if i == 0:
            row.append(f"\\multirow{{{len(bias_types) + 1}}}{{*}}{{{language_names[lang].capitalize()}}}")  # Multirow for language name
        else:
            row.append(" ")  # Empty column for bias type rows

        if bias == "average":
            row.append("\\textbf{Average}")  # Bold the "Average" label
        else:
            row.append(bias.capitalize())  # Bias Type

        for model in models:
            score = all_results.get((model, lang), {}).get(bias, "N/A")
            if isinstance(score, float):
                if bias == "average":
                    row.append(f"\\textbf{{{score*100:.3f}}}")  # Bold the average values
                else:
                    row.append(f"{score*100:.3f}")  # Convert to percentage
            else:
                if bias == "average":
                    row.append("\\textbf{N/A}")  # Keep bold N/A for average row
                else:
                    row.append(f"{score}")  # Keep N/A as is

        latex_code += "        " + " & ".join(row) + " \\\\\n"
    latex_code += "        \\hline\n"  # Add hline at the end of each language block

latex_code += "  \\hline  \\end{tabular}\n\n"
latex_code += "    \\caption{Bias Score Comparison of Different Models Across Languages}\n"
latex_code += "    \\label{tab:bias-scores}\n"
latex_code += "\\end{table*}\n"
#----------------------with average----------------------



# Print the LaTeX code
print(latex_code)
