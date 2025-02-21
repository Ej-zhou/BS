import csv
from openai import OpenAI
import os

from tqdm import tqdm

from googletrans import Translator


# from googletrans import Translator/
import asyncio

async def translate_text(text, target_language="id"):
    if not text.strip():
        return ""  # Return empty string if the input is empty

    translator = Translator()
    try:
        translation = await translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
async def main():
        # Read from zh.csv and create id.csv with Indonesian translations
    with open('zh.csv', 'r', encoding='utf-8') as infile, open('id.csv', 'w', encoding='utf-8', newline='') as outfile:
        reader = list(csv.DictReader(infile))  # Convert to list for tqdm progress tracking
        fieldnames = ['sent_more', 'sent_less', 'stereo_antistereo', 'bias_type', 'sent_more_id', 'sent_less_id']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="Translating sentences", unit="sentence"):
            sent_more_en = row['sent_more']
            sent_less_en = row['sent_less']
            
            # Translate to Indonesian
            sent_more_id = await translate_text(sent_more_en, "id")
            sent_less_id = await translate_text(sent_less_en, "id")

            writer.writerow({
                'sent_more': sent_more_en,
                'sent_less': sent_less_en,
                'stereo_antistereo': row['stereo_antistereo'],
                'bias_type': row['bias_type'],
                'sent_more_id': sent_more_id,
                'sent_less_id': sent_less_id
            })

            print(sent_more_id, sent_less_id)


    print("Translation complete. Saved as id.csv.")

    # text = "Hello, how are you?"
    # translated_text = await translate_text(text, "id")  # "id" is the language code for Indonesian
    # print(translated_text)

asyncio.run(main())

