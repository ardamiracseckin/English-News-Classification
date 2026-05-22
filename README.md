# 📰 BBC News Text Classification with DistilBERT

## 🚀 Overview
This project presents an end-to-end Natural Language Processing (NLP) pipeline that classifies **BBC News** articles into five distinct categories: *Business, Entertainment, Politics, Sport, and Tech*. By fine-tuning the lightweight and highly efficient **DistilBERT** model (`distilbert-base-uncased`), the project demonstrates advanced text classification capabilities using the Hugging Face ecosystem and PyTorch.

## ✨ Project Pipeline & Features

### 1. Exploratory Data Analysis (EDA)
Before training the model, the dataset is thoroughly analyzed to understand its underlying structure:
- **Category Distribution:** Visualizing class balance using Seaborn count plots.
- **Word Count Analysis:** Analyzing the length of the news articles to determine optimal tokenization limits (max length).
- **Word Clouds:** Generating category-specific word clouds to visualize the most frequent and defining keywords for each news class.

### 2. Data Preprocessing & Tokenization
- Handled text tokenization using `DistilBertTokenizerFast`.
- Applied truncation and padding to a `max_length` of 512 tokens.
- Mapped textual categories to numerical labels and created a custom PyTorch `Dataset` class for efficient batch processing.

### 3. Model Fine-Tuning
- Fine-tuned `DistilBertForSequenceClassification` using the **Hugging Face Trainer API**.
- **Training Setup:** 3 Epochs, batch size of 16, evaluating on a 20% validation split.
- **Metrics Tracking:** Dynamically calculated **Weighted F1-Score** and **Accuracy** at the end of each epoch.

### 4. Evaluation & Visualization
The model's performance is comprehensively evaluated:
- Complete **Classification Report** (Precision, Recall, F1-Score for each category).
- **Confusion Matrix:** Plotted as a Seaborn heatmap to visually inspect model predictions against true labels.
- **Learning Curves:** Mapped Validation Loss and Validation Accuracy across epochs using Matplotlib.

### 5. Live Inference (Real-time Testing)
Includes a built-in live testing script. It processes unseen text through the trained model, applies a Softmax function to the logits, and outputs the predicted news category along with a **Confidence Score (%)**.

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch
- **NLP Library:** Hugging Face Transformers (`DistilBERT`, `Trainer`, `TrainingArguments`)
- **Machine Learning:** Scikit-Learn (`train_test_split`, `f1_score`, `confusion_matrix`)
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, WordCloud

## ⚙️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/ardamiracseckin/English-News-Classification-DistilBERT.git](https://github.com/ardamiracseckin/English-News-Classification.git)
   cd English-News-Classification
