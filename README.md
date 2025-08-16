# 🎬 Sentiment Analysis on IMDB Movie Reviews using NLP & Naive Bayes

## 📌 Project Overview
This project focuses on performing **Sentiment Analysis** on the **IMDB Movie Reviews Dataset**.  
The goal is to classify each review as either **Positive (1)** or **Negative (0)** using **Natural Language Processing (NLP)** techniques and a **Naive Bayes classifier**.  

Sentiment analysis helps businesses, researchers, and platforms understand user feedback and improve decision-making.  

---

## 📊 Dataset
- **Source:** IMDB Dataset (50,000 labeled reviews)  
- **Columns:**  
  - `review` → The movie review text  
  - `sentiment` → Positive / Negative  

---

## ⚙️ Steps Implemented
1. **Import Libraries** → pandas, numpy, matplotlib, seaborn, nltk, sklearn  
2. **Load Dataset** → IMDB movie reviews CSV file  
3. **Exploratory Data Analysis (EDA)** → Check duplicates, nulls, class balance, visualizations  
4. **Data Preprocessing** → Clean text (remove punctuation, lowercase, stopwords), encode labels  
5. **Train-Test Split** → 80% training, 20% testing  
6. **Text Vectorization** → TF-IDF (max_features=5000)  
7. **Model Training** → Multinomial Naive Bayes  
8. **Model Evaluation** → Accuracy, Precision, Recall, F1-score, Confusion Matrix  
9. **Important Words** → Extract top positive and negative indicators  
10. **Save Model & Vectorizer** → Pickle for reusability  

---

## 🧪 Results
- **Accuracy:** ~85%  
- **Precision, Recall, F1-score:** ~0.85 for both classes  
- **Top Positive Words:** love, great, amazing, excellent  
- **Top Negative Words:** bad, boring, worst, awful  

✅ The model is well-balanced and performs equally well for both positive and negative reviews.  

---

## 📦 Files in Repository
- `IMDB_Sentiment_Analysis.ipynb` → Full Google Colab Notebook  
- `IMDB Dataset.csv` → Dataset (if permitted to share)  
- `sentiment_model.pkl` → Trained Naive Bayes model  
- `tfidf_vectorizer.pkl` → Saved TF-IDF vectorizer  
- `README.md` → Project documentation  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/IMDB-Sentiment-Analysis-NLP.git
   cd IMDB-Sentiment-Analysis-NLP
2. Install required libraries:
   pip install -r requirements.txt
3. Open the notebook:
   jupyter notebook IMDB_Sentiment_Analysis.ipynb

   
📌 Future Improvements:

Try Logistic Regression, SVM, or Deep Learning models (LSTMs, Transformers)
Use word n-grams for more context
Perform hyperparameter tuning

👨‍💻 Author:

Muhammad Talha Younas
📧 Email: talhayounas696@gmail.com

⭐ Acknowledgement:

Mentor: Ali Mohiuddin Khan
Dataset: IMDB Movie Reviews (50,000 labeled dataset).





   git clone https://github.com/your-username/IMDB-Sentiment-Analysis-NLP.git
   cd IMDB-Sentiment-Analysis-NLP
