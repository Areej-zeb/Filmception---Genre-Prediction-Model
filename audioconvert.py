# convert_summaries_tts.py

import os
import pandas as pd
from googletrans import Translator
from gtts import gTTS
from tqdm.auto import tqdm

# 1) Prepare
INPUT_CSV    = "processed_cleaned_data.csv"
OUTPUT_DIR   = "audio_files"
LANG_SCHEDULE = [
    ("ur",  0, 20),  # first 20 → Urdu
    ("ar", 20, 40),  # next 20 → Arabic
    ("ko", 40, 50),  # next 10 → Korean
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2) Load summaries
df = pd.read_csv(INPUT_CSV, usecols=["plot_summary"])
summaries = df["plot_summary"].astype(str).tolist()

translator = Translator()

def translate_and_tts(text: str, lang: str, out_path: str):
    # do translation
    trans = translator.translate(text, dest=lang).text
    # make speech
    tts = gTTS(text=trans, lang=lang)
    tts.save(out_path)

# 3) Loop through each segment
jobs = []
for lang, start, end in LANG_SCHEDULE:
    for idx in range(start, min(end, len(summaries))):
        jobs.append((idx, summaries[idx], lang))

print(f"Will generate {len(jobs)} files into ./{OUTPUT_DIR}/ …")

for idx, text, lang in tqdm(jobs, desc="Converting"):
    fn = f"summary_{idx+1:03d}_{lang}.mp3"
    out_path = os.path.join(OUTPUT_DIR, fn)
    try:
        translate_and_tts(text, lang, out_path)
    except Exception as e:
        print(f"  ❌ failed idx={idx} lang={lang}: {e}")

print("✅ All done!")
