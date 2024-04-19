from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    # Tokenizare
    words = word_tokenize(text.lower())

    # Eliminăm stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Lemmatizare
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # return string
    return ' '.join(lemmatized_words)


df = pd.read_csv('Resume.csv')

df['Resume_str'] = df['Resume_str'].fillna('')
df['Resume_str'] = df['Resume_str'].apply(preprocess_text)

min_examples = 105
class_counts = df['Category'].value_counts()
weak_classes = class_counts[class_counts < min_examples].index.tolist()

df_filtered = df[~df['Category'].isin(weak_classes)]

class_counts_filtered = df_filtered['Category'].value_counts()
# print("Class counts after filtering:\n", class_counts_filtered)

X_filtered = df_filtered['Resume_str']
y_filtered = df_filtered['Category']

X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered,
                                                                                        test_size=0.2, random_state=42)

tfidf_vectorizer_filtered = TfidfVectorizer(max_features=5000)
X_train_tfidf_filtered = tfidf_vectorizer_filtered.fit_transform(X_train_filtered)
X_test_tfidf_filtered = tfidf_vectorizer_filtered.transform(X_test_filtered)

logistic_regression_filtered = LogisticRegression(max_iter=1000)
logistic_regression_filtered.fit(X_train_tfidf_filtered, y_train_filtered)

y_pred_filtered = logistic_regression_filtered.predict(X_test_tfidf_filtered)

accuracy_filtered = accuracy_score(y_test_filtered, y_pred_filtered)
# print("\nAccuracy for filtered dataset:", accuracy_filtered)
# print("\nClassification Report for filtered dataset:")
# print(classification_report(y_test_filtered, y_pred_filtered))

y_train_pred = logistic_regression_filtered.predict(X_train_tfidf_filtered)
accuracy_train = accuracy_score(y_train_filtered, y_train_pred)


# print("\nAccuracy for training dataset:", accuracy_train)
# print("\nClassification Report for training dataset:")
# print(classification_report(y_train_filtered, y_train_pred))


def preprocess_text_test(text):
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)


def predict_candidates(category):
    engine = create_engine('mysql+mysqlconnector://root:@localhost/HRSystem')
    cv_data = pd.read_sql_query("SELECT * FROM cv WHERE status='pending'", engine)
    relevant_columns = [col for col in cv_data.columns if col != 'id']
    cv_data['Concatenated_Text'] = cv_data[relevant_columns].apply(
        lambda row: ' '.join([str(row[col]) for col in relevant_columns]), axis=1)
    cv_data['Preprocessed_Text'] = cv_data['Concatenated_Text'].fillna('').apply(preprocess_text_test)
    X_cv_data_transformed = tfidf_vectorizer_filtered.transform(cv_data['Preprocessed_Text'])
    y_pred_cv_data = logistic_regression_filtered.predict(X_cv_data_transformed)
    cv_data['Predicted_Category'] = y_pred_cv_data
    desired_category = category
    matching_cv_data = cv_data[cv_data['Predicted_Category'] == desired_category]

    # Creează o listă de dicționare cu informațiile relevante despre candidați
    candidates = []
    for index, row in matching_cv_data.iterrows():
        candidate_info = {
            'name': row['Concatenated_Text'],  # Schimbă 'name' cu coloana relevantă din DataFrame
            # Adaugă și alte informații relevante despre candidați dacă este cazul
        }
        candidates.append(candidate_info)

    return candidates


app = Flask(__name__)
# CORS(app)  # Permit toate originile
CORS(app, origins=['http://localhost:8000'])


@app.route('/recomanda-candidati', methods=['OPTIONS', 'POST'])
def recomanda_candidati():
    try:
        # Primește categoria din cererea POST
        data = request.get_json()
        resume_type = data.get('resume_type')
        print(resume_type)
        # Faceți predicții pe baza categoriei primite
        predicted_candidates = predict_candidates(resume_type)
        print(predicted_candidates)
        # Returnează rezultatele și predicțiile modelului sub formă de JSON
        return jsonify({'candidates': predicted_candidates})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
