# Text-Emotion-Classifier
NLP-based emotion detection model to identify patterns like fear, joy, sadness, and anger in text, designed for emotionally intelligent systems and real-world applications.

Emotion Detection from Text using NLP
Status: Work in Progress

This project focuses on building an NLP-based model that can classify short texts (like tweets or messages) into one of six core emotions: joy, sadness, anger, fear, love, and surprise.

The project is inspired by the goals of the Feminine Intelligence Agency (FIA), which promotes emotional intelligence, trauma-informed communication, and relationship awareness. The idea is to create something useful not only for FIA but also for mental health apps, feedback systems, HR platforms, and any tool that benefits from emotionally aware responses.

Project Purpose

The goal is to detect emotion from text in a way that can support:

AI companions or therapy bots that respond with empathy

Early identification of emotional distress or manipulation

Emotion-aware interfaces for mental health support

Research in behavioral patterns and relationship analysis

Even though FIA is the inspiration, this project is designed to have broader value across industries like healthcare, education, and communication platforms.

Current Progress

Dataset loaded and structured

Column cleanup and label mapping complete

Emotion distribution visualized with bar charts

Tweet length distribution analyzed using histograms

Text cleaned: lowercased, URLs and symbols removed

Cleaned version of each tweet stored in a new column

Upcoming Steps

Apply TF-IDF or embedding-based feature extraction

Train initial models like Logistic Regression and SVM

Explore fine-tuning BERT or other transformer models

Evaluate model performance using accuracy and F1 score

Build a basic web app (Streamlit or Flask) for input-based predictions

Tools and Libraries Used

Python 3.10

pandas and NumPy

matplotlib for visualization

re (regex) for cleaning text

scikit-learn for modeling

Hugging Face Transformers (planned for deep learning)

Planned Folder Structure

data/ for raw and cleaned CSVs

notebooks/ for analysis and model development

models/ to store trained models

app/ for building a small web interface

utils/ for reusable code (like cleaning functions)

Contact

This project is being developed independently and inspired by FIA's mission. If you'd like to collaborate, suggest improvements, or discuss potential use cases, feel free to reach out via GitHub Issues.
