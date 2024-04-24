from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)


df = pd.read_csv('Resume.csv')

df['Resume_str'] = df['Resume_str'].fillna('')
df['Resume_str'] = df['Resume_str'].apply(preprocess_text)

min_examples = 105
class_counts = df['Category'].value_counts()
weak_classes = class_counts[class_counts < min_examples].index.tolist()
df_filtered = df[~df['Category'].isin(weak_classes)]

X_filtered = df_filtered['Resume_str']
y_filtered = df_filtered['Category']

X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered,
                                                                                        test_size=0.2, random_state=42)
tfidf_vectorizer_filtered = TfidfVectorizer(max_features=5000)
X_train_tfidf_filtered = tfidf_vectorizer_filtered.fit_transform(X_train_filtered)
X_test_tfidf_filtered = tfidf_vectorizer_filtered.transform(X_test_filtered)

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train_tfidf_filtered, y_train_filtered)

y_pred_filtered_random_forest = random_forest_classifier.predict(X_test_tfidf_filtered)

accuracy_filtered_random_forest = accuracy_score(y_test_filtered, y_pred_filtered_random_forest)
print("\nAccuracy for filtered dataset using Random Forest:", accuracy_filtered_random_forest)
print("\nClassification Report for filtered dataset using Random Forest:")
print(classification_report(y_test_filtered, y_pred_filtered_random_forest))

y_train_pred_random_forest = random_forest_classifier.predict(X_train_tfidf_filtered)
accuracy_train_random_forest = accuracy_score(y_train_filtered, y_train_pred_random_forest)
print("\nAccuracy for training dataset using Random Forest:", accuracy_train_random_forest)
print("\nClassification Report for training dataset using Random Forest:")
print(classification_report(y_train_filtered, y_train_pred_random_forest))

def predict_candidates(category):
    engine = create_engine('mysql+mysqlconnector://root:@localhost/HRSystem')
    cv_data = pd.read_sql_query("SELECT * FROM cv WHERE status='pending'", engine)
    relevant_columns = [col for col in cv_data.columns if col != 'id']
    cv_data['Concatenated_Text'] = cv_data[relevant_columns].apply(
        lambda row: ' '.join([str(row[col]) for col in relevant_columns]), axis=1)
    cv_data['Preprocessed_Text'] = cv_data['Concatenated_Text'].fillna('').apply(preprocess_text)
    X_cv_data_transformed = tfidf_vectorizer_filtered.transform(cv_data['Preprocessed_Text'])
    y_pred_cv_data = random_forest_classifier.predict(X_cv_data_transformed)
    cv_data['Predicted_Category'] = y_pred_cv_data
    desired_category = category
    matching_cv_data = cv_data[cv_data['Predicted_Category'] == desired_category]

    candidates = []
    for index, row in matching_cv_data.iterrows():
        candidate_info = {
            'id': row['id'],
            'name': row['name'],
            'email': row['email'],
            # 'street_address': row['street_address'],
            # 'postal_code': row['postal_code'],
            # 'city': row['city'],
            # 'county': row['county'],
            # 'country': row['country'],
            'phone': row['phone'],
            # 'birthdate': row['birthdate'],
            # 'education_institution_1': row['education_institution_1'],
            # 'education_major_1': row['education_major_1'],
            # 'education_degree_1': row['education_degree_1'],
            # 'education_graduation_year_1': row['education_graduation_year_1'],
            # 'education_institution_2': row['education_institution_2'],
            # 'education_major_2': row['education_major_2'],
            # 'education_degree_2': row['education_degree_2'],
            # 'education_graduation_year_2': row['education_graduation_year_2'],
            # 'experience_company_1': row['experience_company_1'],
            # 'experience_position_1': row['experience_position_1'],
            # 'experience_period_1': row['experience_period_1'],
            # 'experience_responsibilities_1': row['experience_responsibilities_1'],
            # 'experience_company_2': row['experience_company_2'],
            # 'experience_position_2': row['experience_position_2'],
            # 'experience_period_2': row['experience_period_2'],
            # 'experience_responsibilities_2': row['experience_responsibilities_2'],
            # 'language_1': row['language_1'],
            # 'language_level_1': row['language_level_1'],
            # 'language_2': row['language_2'],
            # 'language_level_2': row['language_level_2'],
            # 'technical_skills': row['technical_skills'],
            # 'soft_skills': row['soft_skills'],
            # 'status': row['status']
        }
        candidates.append(candidate_info)

    return candidates


app = Flask(__name__)
# CORS(app)  # Permit toate originile
CORS(app, origins=['http://localhost:8000'])


@app.route('/recomanda-candidati', methods=['OPTIONS', 'POST'])
def recomanda_candidati():
    try:
        data = request.get_json()
        resume_type = data.get('resume_type')
        print(resume_type)
        predicted_candidates = predict_candidates(resume_type)
        print(predicted_candidates)
        return jsonify({'candidates': predicted_candidates})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
