import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset (you can expand later)
data = {
    "text": [
        "Government launches new scheme for farmers",
        "Aliens landed in India yesterday",
        "Stock market hits record high",
        "Fake cure for cancer discovered online"
    ],
    "label": [1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)

# Convert text to vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

mlflow.set_experiment("Fake_News_Detection")

models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)

        # Log
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, name)

        print(f"{name} Accuracy:", acc)