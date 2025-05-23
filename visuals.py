import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from itertools import combinations
import tkinter as tk
from PIL import Image, ImageTk

# Create output folder
os.makedirs('visuals', exist_ok=True)

# Load processed data
df = pd.read_csv('processed_cleaned_data.csv')

# Determine summary column
summary_col = 'clean_summary' if 'clean_summary' in df.columns else 'plot_summary'

# 1) Histogram of summary lengths
summary_lens = df[summary_col].astype(str).apply(lambda x: len(x.split()))
plt.figure()
plt.hist(summary_lens, bins=50)
plt.title('Distribution of Summary Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/summary_length_histogram.png')
plt.close()

# 2) Genre frequency top 20
# Parse cleaned_genres if string
if df['cleaned_genres'].dtype == object:
    df['cleaned_genres'] = df['cleaned_genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)
from collections import Counter
all_genres = Counter(g for sub in df['cleaned_genres'] for g in sub)
top20 = all_genres.most_common(20)
genres, counts = zip(*top20)
plt.figure()
plt.barh(genres[::-1], counts[::-1])
plt.title('Top 20 Most Frequent Genres')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('visuals/genre_frequency_top20.png')
plt.close()

# 3) Average summary length by genre (top 10)
avg_len = {}
for genre, _ in top20[:10]:
    lens = summary_lens[df['cleaned_genres'].apply(lambda lst: genre in lst)]
    avg_len[genre] = lens.mean()
b = sorted(avg_len.items(), key=lambda x: x[1], reverse=True)
genres_avg, lengths_avg = zip(*b)
plt.figure()
plt.barh(genres_avg[::-1], lengths_avg[::-1])
plt.title('Avg Summary Length by Genre (Top 10)')
plt.xlabel('Avg # of Words')
plt.tight_layout()
plt.savefig('visuals/avg_summary_length_by_genre.png')
plt.close()

# 4) Histogram of number of genres per movie
num_genres = df['cleaned_genres'].apply(len)
plt.figure()
plt.hist(num_genres, bins=range(1, max(num_genres)+2))
plt.title('Number of Genres per Movie')
plt.xlabel('Count of Genres')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.savefig('visuals/num_genres_histogram.png')
plt.close()

# 5) Genre co-occurrence network (thresholded)
cooc = Counter()
for genres in df['cleaned_genres']:
    for a, b in combinations(sorted(genres), 2):
        cooc[(a,b)] += 1
G = nx.Graph()
for (a,b), cnt in cooc.items():
    if cnt > 100:  # only strong co-occurrence
        G.add_edge(a, b, weight=cnt)
plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k=0.15)
nx.draw_networkx_nodes(G, pos, node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title('Genre Co-occurrence (edges > 100)')
plt.axis('off')
plt.tight_layout()
plt.savefig('visuals/genre_cooccurrence_network.png')
plt.close()

# 6) WordClouds for top 5 genres
for genre, _ in top20[:5]:
    texts = " ".join(df.loc[df['cleaned_genres'].apply(lambda lst: genre in lst), summary_col].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(texts)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for {genre}')
    plt.tight_layout()
    fname = f"visuals/wordcloud_{genre.replace(' ','_')}.png"
    plt.savefig(fname)
    plt.close()

# --- Popup GUI to display visuals ---
root = tk.Tk()
root.title('Data Visualizations')
frame = tk.Frame(root)
frame.pack(fill='both', expand=True)
canvas = tk.Canvas(frame)
scroll_y = tk.Scrollbar(frame, orient='vertical', command=canvas.yview)

inner = tk.Frame(canvas)

# load and display each image
images = []
for fname in sorted(os.listdir('visuals')):
    if fname.endswith('.png'):
        path = os.path.join('visuals', fname)
        img = Image.open(path)
        img.thumbnail((600,400))
        photo = ImageTk.PhotoImage(img)
        images.append(photo)
        lbl = tk.Label(inner, image=photo)
        lbl.pack(pady=10)

canvas.create_window((0,0), window=inner, anchor='nw')
canvas.update_idletasks()
canvas.configure(scrollregion=canvas.bbox('all'),
                 yscrollcommand=scroll_y.set)
canvas.pack(fill='both', expand=True, side='left')
scroll_y.pack(fill='y', side='right')
root.mainloop()
