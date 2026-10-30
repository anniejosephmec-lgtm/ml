"""Q7: Compare Multinomial and Bernoulli Naive Bayes on 20 Newsgroups."""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
vectorizer = CountVectorizer(stop_words="english", max_features=20000, binary=False)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

for name, model in [
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Bernoulli Naive Bayes", BernoulliNB())
]:
    model.fit(X_train, train.target)
    pred = model.predict(X_test)
    print("\\n", name)
    print("Accuracy:", round(accuracy_score(test.target, pred), 3))
    print("Macro F1:", round(f1_score(test.target, pred, average="macro"), 3))
