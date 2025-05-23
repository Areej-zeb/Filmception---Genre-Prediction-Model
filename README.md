
# 🎬 Filmception: AI-Powered Movie Genre Predictor & Audio Converter

**Filmception** is a multilingual, interactive AI application that allows users to input a movie summary, get audio translations in different languages, and predict the movie's genre(s) using a fine-tuned BERT model. This project integrates modern natural language processing (NLP), machine learning, and user interface design.

---

## 📌 Features

- 🧠 **Genre Classification:** Predicts one or more genres using a fine-tuned BERT model.
- 🌐 **Multilingual Support:** Translate summaries into **Urdu**, **Arabic**, or **Korean** using Google Translate.
- 🔊 **Text-to-Speech Conversion:** Converts translated summaries into audio with **gTTS**.
- 🎛️ **Interactive GUI:** Built with **Tkinter**, includes audio playback controls (Play, Pause, Rewind).
- 📈 **Visual Analytics:** Genre distributions, summary lengths, co-occurrence heatmaps, word clouds, and more.

---

## 📂 Project Structure

```
Filmception/
├── app.py                   # Main GUI application
├── preprocess_data.py       # Data cleaning and preprocessing
├── train_model.py           # Fine-tuning BERT for genre classification
├── visuals.py               # Data visualizations (histograms, networks, word clouds)
├── processed_cleaned_data.csv
├── genre_prediction_model_bert/  # Saved BERT model and tokenizer
├── visuals/                 # Generated graphs and images
├── audio_files/             # Translated TTS output files
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- **Python 3**
- **Hugging Face Transformers (BERT)**
- **NLTK** for text preprocessing
- **scikit-learn** for MultiLabelBinarizer
- **gTTS** and **pygame** for audio
- **Tkinter** for UI
- **Matplotlib**, **WordCloud**, **NetworkX** for visualizations
- **Googletrans** for translation

---

## 🧪 How It Works

1. **Preprocessing**: `preprocess_data.py` loads and cleans the CMU Movie Summary dataset.
2. **Training**: `train_model.py` fine-tunes BERT for multi-label genre classification.
3. **Visualization**: `visuals.py` analyzes and visualizes genre distributions and trends.
4. **Application**: `app.py` runs a GUI where users:
   - Enter a movie summary.
   - Translate and hear the summary.
   - Choose single/multi-genre classification.
   - View the predicted genres.

---

## 🧠 AI Concepts Applied

- **Multi-label classification** using BERT.
- **Tokenization, lemmatization, and vectorization** of text data.
- **Transfer learning** from `bert-base-uncased`.
- **Evaluation metrics**: Accuracy, precision, recall, F1-score.

---

## 📊 Sample Visuals

- Distribution of Summary Lengths
- Top 20 Frequent Genres
- Co-occurrence Heatmap & Network
- WordClouds per Genre

All are auto-generated and saved in the `/visuals` folder.

---

## 🖥️ Demo

1. Launch the GUI with:
   ```bash
   python app.py
   ```
2. Input a movie summary.
3. Select a translation language and convert to audio.
4. Choose classification mode: **Single** (Top-1) or **Multi** (Threshold-based).
5. View genre prediction and play audio.

---

## 📦 Installation

1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download [CMU Movie Summary Dataset](https://www.kaggle.com/datasets/msafi04/movies-genre-dataset-cmu-moviesummary) and place it in the correct folder.

---


## 📚 References

- CMU Movie Summary Corpus: [Kaggle Dataset](https://www.kaggle.com/datasets/msafi04/movies-genre-dataset-cmu-moviesummary)
- Hugging Face Transformers
- Google Translate API via `googletrans`
- `gTTS` for text-to-speech
