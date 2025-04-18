# Instagram Topic Modeling Pipeline

This repository contains a pipeline for collecting and analyzing Instagram social media content, specifically focused on **topic modeling** of media (images and videos) and comments.

---

## 📁 Repository Structure

### `database_creation/`
This folder contains scripts and data used to collect and build the raw Instagram dataset.

- `database_creation1.ipynb` — Notebook for scraping and building datasets using `instagram-scraper`.
- `dataset_instagram-scraper_*.json` — JSON files containing scraped Instagram post data from the brand's , including metadata and media links.
- `dataset_indirect_posts_instagram-scraper_*.json` — JSON files containing scraped Instagram post data , including metadata and media links.

---

### `topic_modelling/`
This folder contains scripts and outputs related to **topic modeling** on both textual comments and visual media.

- `topic_modeling.py` — Script for modeling topics from text (captions from indirect posts (not from the brand) or comments).
- `media_topic_modeling.py` — Script for extracting keyframes from videos, embedding images using CLIP, and clustering them into visual topics.
- `media_topic_modeling.log` — Log file capturing the media modeling process and progress.
- `comment_topics.csv` — Output file containing clustered comment topics.
- `media_topics.csv` — Output file containing clustered media topic assignments.

---

## 🔧 Requirements

- Python 3.8+
- Packages:
  - `opencv-python`
  - `torch`
  - `clip-by-openai`
  - `Pillow`
  - `scikit-learn`
  - `numpy`
  - `tqdm`
  - `pandas`
