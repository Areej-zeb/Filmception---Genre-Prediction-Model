import os, glob, tkinter as tk
from tkinter import ttk, messagebox
from googletrans import Translator
from gtts import gTTS
import pygame
import joblib
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# ─── 1) Setup ────────────────────────────────────────────────────────────────
pygame.mixer.init()
translator = Translator()

# load your fine-tuned BERT
model = BertForSequenceClassification.from_pretrained(
    "./genre_prediction_model_bert",
    problem_type="multi_label_classification"
)
tokenizer = BertTokenizerFast.from_pretrained("./genre_prediction_model_bert")
model.eval()

# load MultiLabelBinarizer only for label names
mlb = joblib.load("mlb_encoder.joblib")
genre_labels = list(mlb.classes_)
# simple per-label thresholds
thresholds = {g:0.3 for g in genre_labels}
thresholds.update({"Drama":0.25,"Comedy":0.25,"Science Fiction":0.35,"Horror":0.35})

def predict_top_n(text,n=1):
    enc = tokenizer(text, return_tensors="pt",
                    padding=True,truncation=True,max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    ix = probs.argsort()[-n:][::-1]
    return [genre_labels[i] for i in ix]

def predict_multi(text):
    enc = tokenizer(text, return_tensors="pt",
                    padding=True,truncation=True,max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    return [g for i,g in enumerate(genre_labels) if probs[i]>=thresholds[g]]

def next_mp3_filename(lang_code):
    os.makedirs("audio_files",exist_ok=True)
    existing = glob.glob(f"audio_files/movie_summary_{lang_code}_*.mp3")
    if not existing: return f"audio_files/movie_summary_{lang_code}_1.mp3"
    idxs = [int(f.rsplit("_",1)[1].split(".")[0]) for f in existing]
    return f"audio_files/movie_summary_{lang_code}_{max(idxs)+1}.mp3"

current_audio = None
is_paused = False

def generate_audio():
    global current_audio, is_paused
    s = txt_summary.get("1.0","end").strip()
    if not s:
        messagebox.showwarning("Missing","Enter a summary first.")
        return
    code = langs[lang_var.get()]
    # translate first
    trans = translator.translate(s,dest=code).text
    txt_trans.delete("1.0","end")
    txt_trans.insert("1.0",trans)
    # TTS
    fn = next_mp3_filename(code)
    tts = gTTS(text=trans, lang=code)
    tts.save(fn)
    current_audio = fn
    is_paused = False
    pygame.mixer.music.load(fn)
    pygame.mixer.music.play()
    btn_play.config(state="disabled")
    btn_pause.config(state="normal")
    btn_rewind.config(state="normal")
    messagebox.showinfo("Saved","Audio saved as:\n"+fn)

def play_audio():
    global is_paused
    if not current_audio:
        messagebox.showwarning("No audio","Generate audio first.")
        return
    if is_paused:
        pygame.mixer.music.unpause()
        is_paused = False
    else:
        pygame.mixer.music.load(current_audio)
        pygame.mixer.music.play()
    btn_play.config(state="disabled")
    btn_pause.config(state="normal")

def pause_audio():
    global is_paused
    if not current_audio: return
    pygame.mixer.music.pause()
    is_paused = True
    btn_play.config(state="normal")
    btn_pause.config(state="disabled")

def rewind_audio():
    if not current_audio: return
    pygame.mixer.music.stop()
    pygame.mixer.music.play()
    btn_play.config(state="disabled")
    btn_pause.config(state="normal")

def do_predict():
    s = txt_summary.get("1.0","end").strip()
    if not s:
        messagebox.showwarning("Missing","Enter a summary first.")
        return
    if mode_var.get()=="Single":
        preds = predict_top_n(s,n=1)
    else:
        preds = predict_multi(s)
    lbl_gen.config(text="Predicted: "+(", ".join(preds) if preds else "None"))

# ─── 2) Build GUI ───────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Movie Genre & Audio")
root.configure(bg="#2E3B4E")  # dark slate

FG    = "#ECEFF4"
INPUT_BG = "#3B4252"
BTN_BG = "#5E81AC"
BTN_FG = "#ECEFF4"

tk.Label(root, text="🎬 Movie Genre & Audio", font=("Helvetica",18,"bold"),
         bg="#2E3B4E", fg=FG).pack(pady=(20,10))

# summary
tk.Label(root, text="Enter movie summary:", font=("Arial",12),
         bg="#2E3B4E", fg=FG).pack(anchor="w", padx=30)
txt_summary = tk.Text(root, height=6, width=60, bg=INPUT_BG, fg=FG,
                      insertbackground=FG, wrap="word")
txt_summary.pack(padx=30, pady=(0,20))

# translation frame
frame_trans = tk.Frame(root, bg="#2E3B4E")
frame_trans.pack(pady=(0,10))
tk.Label(frame_trans, text="Language:", bg="#2E3B4E", fg=FG).pack(side="left")
langs = {"Urdu":"ur","Arabic":"ar","Korean":"ko"}
lang_var = tk.StringVar(value="Urdu")
ttk.Combobox(frame_trans, values=list(langs.keys()),
             textvariable=lang_var, state="readonly", width=10).pack(side="left",padx=5)
tk.Button(frame_trans, text="Translate & Audio", command=generate_audio,
          bg=BTN_BG, fg=BTN_FG, font=("Arial",10,"bold")).pack(side="left",padx=10)

# translated text
tk.Label(root, text="Translated summary:", font=("Arial",12),
         bg="#2E3B4E", fg=FG).pack(anchor="w", padx=30)
txt_trans = tk.Text(root, height=4, width=60, bg=INPUT_BG, fg=FG,
                    insertbackground=FG, wrap="word")
txt_trans.pack(padx=30, pady=(0,15))

# audio controls
frame_audio = tk.Frame(root, bg="#2E3B4E")
frame_audio.pack(pady=(0,15))
btn_play = tk.Button(frame_audio, text="▶ Play",   command=play_audio,
                     bg=BTN_BG, fg=BTN_FG, font=("Arial",10,"bold"), state="disabled")
btn_pause= tk.Button(frame_audio, text="❚❚ Pause", command=pause_audio,
                     bg=BTN_BG, fg=BTN_FG, font=("Arial",10,"bold"), state="disabled")
btn_rewind=tk.Button(frame_audio, text="« Rewind",command=rewind_audio,
                     bg=BTN_BG, fg=BTN_FG, font=("Arial",10,"bold"), state="disabled")
for b in (btn_play,btn_pause,btn_rewind): b.pack(side="left", padx=5)

# classification mode
mode_var = tk.StringVar(value="Single")
frame_mode = tk.Frame(root, bg="#2E3B4E")
frame_mode.pack(pady=(0,10))
tk.Label(frame_mode, text="Classification:", bg="#2E3B4E", fg=FG).pack(side="left")
tk.Radiobutton(frame_mode, text="Single", variable=mode_var, value="Single",
               bg="#2E3B4E", fg=FG, selectcolor=INPUT_BG).pack(side="left", padx=5)
tk.Radiobutton(frame_mode, text="Multi", variable=mode_var, value="Multi",
               bg="#2E3B4E", fg=FG, selectcolor=INPUT_BG).pack(side="left", padx=5)
tk.Button(frame_mode, text="Predict Genres", command=do_predict,
          bg=BTN_BG, fg=BTN_FG, font=("Arial",10,"bold")).pack(side="left", padx=20)

# result label
lbl_gen = tk.Label(root, text="Predicted: —", font=("Arial",11),
                   bg="#2E3B4E", fg=FG)
lbl_gen.pack(pady=(0,20))

# exit
tk.Button(root, text="Exit", command=root.destroy,
          bg="#BF616A", fg=BTN_FG, font=("Arial",10,"bold"),
          width=10).pack(pady=(0,30))

root.mainloop()
