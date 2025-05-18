# spam_detector
The Spam Classification with Random Forest Project builds a machine learning model to detect spam messages using a Random Forest classifier. It uses NLP techniques to process and extract features from text data, enabling accurate classification of messages as spam or non-spam (ham), thereby enhancing user experience on messaging platforms.

Objectives

Build an end-to-end NLP pipeline for text classification using Python.
Preprocess unstructured text to remove noise and standardize input.
Apply feature extraction to convert text into numerical features suitable for Random Forest.
Train and evaluate a Random Forest classifier for accurate spam detection.
Provide a reusable framework for Random Forest-based text classification tasks.

Methodology
The project follows a structured machine learning workflow:
Data Loading:
Dataset: A CSV file (spam.csv) containing text messages and their labels ("spam" or "ham").
Tools: pandas for loading and extracting message text (features) and labels.
Data Preprocessing:
Text Cleaning: Remove non-alphabetic characters using regular expressions, convert text to lowercase, tokenize into words with NLTK’s word_tokenize, remove English stop words (via stopwords), and lemmatize words to their base form using WordNetLemmatizer.
Label Encoding: Convert categorical labels ("spam", "ham") to numerical values (0, 1) with scikit-learn’s LabelEncoder.
Data Splitting:
Split the dataset into training (75%) and testing (25%) sets using train_test_split with random_state=0 for reproducibility.
Feature Extraction:
Transform preprocessed text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) with TfidfVectorizer from scikit-learn.
Generate sparse matrices (X_train_tfidf, X_test_tfidf) capturing word importance, optimized for Random Forest’s high-dimensional input handling.

Model Training:
Train a Random Forest Classifier (RandomForestClassifier) with 1000 trees and entropy criterion on the TF-IDF features.
Random Forest was selected for its ensemble approach, robustness to overfitting, and ability to handle sparse, high-dimensional text data effectively.

Prediction and Evaluation:
Predict labels for test data and a sample input text (e.g., "Hello mate, you are the winner...").

Evaluate performance using accuracy and confusion matrix from scikit-learn.metrics.
The Random Forest model demonstrates high accuracy, reflecting its effectiveness in spam classification.

Technologies Used
Programming Language: Python 3.x

Libraries:
pandas: Data loading and manipulation.
numpy: Numerical operations for array handling.
nltk: Text preprocessing (tokenization, stop word removal, lemmatization).
scikit-learn: Feature extraction (TF-IDF), data splitting, label encoding, Random Forest classifier, and evaluation metrics.
Dataset: spam.csv (e.g., from UCI Machine Learning Repository or Kaggle).
Environment: Jupyter Notebook (Google Colab).

Key Features
Effective Preprocessing: Cleans text by removing noise, standardizing case, and reducing vocabulary through stop word removal and lemmatization, ensuring high-quality input for Random Forest.
Robust Feature Extraction: TF-IDF vectorization creates sparse, numerical representations of text, well-suited for Random Forest’s ensemble
