#!/bin/bash

# ------------------------------RUN MULTIPLE------------------------------
# Define the input files
# INPUT_FILES=("en.csv" "zh.csv" "ru.csv" "th.csv" "id.csv")  # Add your actual filenames here
INPUT_FILES=("en.csv" "zh.csv" "ru.csv" "th.csv" "id.csv")  # Add your actual filenames here
LM_MODEL="llama3"  # Options: bert, roberta, albert 
OUTPUT_DIR="output/${LM_MODEL}" # Define output directory

 

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over input files
for FILE in "${INPUT_FILES[@]}"; do
    INPUT_PATH="Feb/$FILE"
    OUTPUT_PATH="$OUTPUT_DIR/$FILE"

    echo "Processing $FILE with model $LM_MODEL..."
    python Feb/metric.py --input_file "$INPUT_PATH" --lm_model "$LM_MODEL" --output_file "$OUTPUT_PATH"
done

echo "Processing completed!"



# ------------------------------RUN SINGLE------------------------------
# # Define input arguments
# INPUT_FILE="Feb/en.csv"
# LM_MODEL="mbert"  # Options: bert, roberta, albert
# OUTPUT_FILE="output.csv"

# # Run the Python script
# python Feb/metric.py --input_file "$INPUT_FILE" --lm_model "$LM_MODEL" --output_file "$OUTPUT_FILE"
