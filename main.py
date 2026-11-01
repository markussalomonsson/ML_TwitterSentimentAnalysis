import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']
print(df.head())

# collapse dataset into binary classes, negative 0 or positvive 1
df = df[df.polarity != 2]

df['polarity'] = df['polarity'].map({0: 0, 4: 1})

def clean_text(text):
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['polarity'],
    test_size=0.2,
    random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))