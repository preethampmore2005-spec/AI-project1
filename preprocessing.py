from imports import *

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Features and target
X = df['message']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

X_tfidf = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)