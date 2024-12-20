import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import re

# 定義路徑
data_identification_path = "/Users/shan/Desktop/資料探勘/com/data_identification.csv"
emotion_path = "/Users/shan/Desktop/資料探勘/com/emotion.csv"
tweets_path = "/Users/shan/Desktop/資料探勘/com/tweets_DM.json"

# Step 1: Load Data
# Load Multi-line JSON file
tweets = []
with open(tweets_path, 'r', encoding='utf-8') as file:
    for line in file:
        tweets.append(json.loads(line.strip()))

# Convert to DataFrame
tweets_df = pd.DataFrame([
    {"tweet_id": tweet["_source"]["tweet"]["tweet_id"], 
     "text": tweet["_source"]["tweet"]["text"]}
    for tweet in tweets
])

# Load other CSV files
data_id = pd.read_csv(data_identification_path)
emotion = pd.read_csv(emotion_path)

# Step 2: Merge and Clean
merged = data_id.merge(tweets_df, left_on="tweet_id", right_on="tweet_id", how="left")
merged = merged.merge(emotion, on="tweet_id", how="left")

# Clean text: remove <LH>, special characters, and convert to lowercase
def clean_text(text):
    text = re.sub(r"<LH>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text.strip()

merged["cleaned_text"] = merged["text"].fillna("").apply(clean_text)

# Step 3: Split Data
train_data = merged[merged["identification"] == "train"].dropna(subset=["emotion"])
test_data = merged[merged["identification"] == "test"]

# Encode labels into integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["emotion"])  # Convert to numeric
X_train = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english').fit_transform(train_data["cleaned_text"])
X_test = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english').fit_transform(test_data["cleaned_text"])

# Step 4: Model Training with RandomizedSearchCV
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    scoring='f1_macro',
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)

# Best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# Step 5: Predict Emotions for Test Data
y_test_pred = best_model.predict(X_test)
test_data["emotion"] = label_encoder.inverse_transform(y_test_pred)  # Convert back to original labels

# Step 6: Create Submission File
submission = test_data[["tweet_id", "emotion"]].rename(columns={"tweet_id": "id"})
submission_path = "/Users/shan/Desktop/資料探勘/com/submission_fixed.csv"
submission.to_csv(submission_path, index=False)

print(f"Fixed submission file created at {submission_path}")
