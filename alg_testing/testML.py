import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sqlalchemy import create_engine

engine = create_engine('mysql+mysqlconnector://root:@localhost/HRSystem')

cv_data = pd.read_sql_query("SELECT * FROM cv WHERE status='analyzed'", engine)
job_data = pd.read_sql_query("SELECT * FROM jobs", engine)

pivot_data = pd.read_sql_query("SELECT cv_id, job_id, employee FROM job_has_cv", engine)

merged_data = pd.merge(pivot_data, cv_data, left_on='cv_id', right_on='id')
merged_data = pd.merge(merged_data, job_data, left_on='job_id', right_on='id')

merged_data['accepted'] = merged_data['employee']
merged_data['accepted'] = merged_data['accepted'].astype(int)

vectorizer = TfidfVectorizer(stop_words='english')

merged_data = merged_data.dropna(
    subset=['technical_skills', 'soft_skills', 'experience_position_1', 'experience_responsibilities_1'])
X_cv = vectorizer.fit_transform(merged_data['technical_skills'] + ' ' + merged_data['soft_skills'] + ' ' + merged_data[
    'experience_position_1'] + ' ' + merged_data['experience_responsibilities_1'])
y = merged_data['accepted']

X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size=0.2, random_state=42)

class_0 = merged_data[merged_data['accepted'] == 0]
class_1 = merged_data[merged_data['accepted'] == 1]
class_1_oversampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
oversampled_train_data = pd.concat([class_0, class_1_oversampled])
X_train_oversampled = vectorizer.transform(oversampled_train_data['technical_skills'] + ' ' + oversampled_train_data['soft_skills'] + ' ' + oversampled_train_data['experience_position_1'] + ' ' + oversampled_train_data['experience_responsibilities_1'])
y_train_oversampled = oversampled_train_data['accepted']

model = MultinomialNB()
model.fit(X_train_oversampled, y_train_oversampled)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

y_train_pred_oversampled = model.predict(X_train_oversampled)
train_accuracy_oversampled = accuracy_score(y_train_oversampled, y_train_pred_oversampled)
train_precision_oversampled = precision_score(y_train_oversampled, y_train_pred_oversampled)
train_recall_oversampled = recall_score(y_train_oversampled, y_train_pred_oversampled)
train_f1_oversampled = f1_score(y_train_oversampled, y_train_pred_oversampled)

print("Performanța pe datele de antrenament după over-sampling:")
print("Acuratețe:", train_accuracy_oversampled)
print("Precizie:", train_precision_oversampled)
print("Recuperare:", train_recall_oversampled)
print("Scor F1:", train_f1_oversampled)

print("\nPerformanța pe datele de testare:")
print("Acuratețe:", accuracy)
print("Precizie:", precision)
print("Recuperare:", recall)
print("Scor F1:", f1)


job_id_to_generate_for = 55

job_to_generate_for_query = f"SELECT * FROM jobs WHERE id={job_id_to_generate_for}"
job_to_generate_for = pd.read_sql_query(job_to_generate_for_query, engine)

if not job_to_generate_for.empty:

    job_tech_skills = job_to_generate_for['requirements_skills_technical'].iloc[0]
    job_soft_skills = job_to_generate_for['requirements_skills_soft'].iloc[0]
    job_title = job_to_generate_for['title'].iloc[0]

    X_job_skills = vectorizer.transform([job_tech_skills + ' ' + job_soft_skills + ' ' + job_title])

    job_prediction = model.predict(X_job_skills)

    new_cv_data = pd.read_sql_query("SELECT * FROM cv WHERE status='pending'", engine)

    if not new_cv_data.empty:
        for index, cv_row in new_cv_data.iterrows():
            cv_technical_skills = cv_row['technical_skills']
            cv_soft_skills = cv_row['soft_skills']
            cv_experience_position_1 = cv_row['experience_position_1']
            cv_experience_responsibilities_1 = cv_row['experience_responsibilities_1']

            if pd.notnull(cv_technical_skills) and pd.notnull(cv_soft_skills) and pd.notnull(
                    cv_experience_position_1) and pd.notnull(cv_experience_responsibilities_1):
                X_cv = vectorizer.transform([
                                                cv_technical_skills + ' ' + cv_soft_skills + ' ' + cv_experience_position_1 + ' ' + cv_experience_responsibilities_1])

                cv_prediction = model.predict(X_cv)

                if cv_prediction[0] == 1 and job_prediction[0] == 1:
                    print(
                        f"The CV '{cv_row['name'] + cv_row['technical_skills']}' is suitable for the job '{job_to_generate_for['title'].iloc[0]}'!")
            else:
                print("One or both of the skills fields are None or empty. Skipping CV processing.")