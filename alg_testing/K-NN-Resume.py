import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Descărcarea resurselor NLTK (dacă nu sunt deja descărcate)
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

def preprocess_text_test(text):
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)

df = pd.read_csv('../Resume.csv')

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

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_tfidf_filtered, y_train_filtered)

y_pred_filtered_knn = knn_classifier.predict(X_test_tfidf_filtered)

accuracy_filtered_knn = accuracy_score(y_test_filtered, y_pred_filtered_knn)
print("\nAccuracy for filtered dataset using KNN:", accuracy_filtered_knn)
print("\nClassification Report for filtered dataset using KNN:")
print(classification_report(y_test_filtered, y_pred_filtered_knn))

y_train_pred_knn = knn_classifier.predict(X_train_tfidf_filtered)
accuracy_train_knn = accuracy_score(y_train_filtered, y_train_pred_knn)
print("\nAccuracy for training dataset using KNN:", accuracy_train_knn)
print("\nClassification Report for training dataset using KNN:")
print(classification_report(y_train_filtered, y_train_pred_knn))

engine = create_engine('mysql+mysqlconnector://root:@localhost/HRSystem')

cv_data = pd.read_sql_query("SELECT * FROM cv WHERE status='pending'", engine)

relevant_columns = [col for col in cv_data.columns if col != 'id']
cv_data['Concatenated_Text'] = cv_data[relevant_columns].apply(
    lambda row: ' '.join([str(row[col]) for col in relevant_columns]), axis=1)
cv_data['Preprocessed_Text'] = cv_data['Concatenated_Text'].fillna('').apply(preprocess_text_test)

X_cv_data_transformed_knn = tfidf_vectorizer_filtered.transform(cv_data['Preprocessed_Text'])

y_pred_cv_data_knn = knn_classifier.predict(X_cv_data_transformed_knn)

cv_data['Predicted_Category_KNN'] = y_pred_cv_data_knn

desired_category = 'BANKING'
matching_cv_data_knn = cv_data[cv_data['Predicted_Category_KNN'] == desired_category]

for index, row in matching_cv_data_knn.iterrows():
    print(row['Concatenated_Text'])
