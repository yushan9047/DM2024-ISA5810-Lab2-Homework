import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

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
merged["cleaned_text"] = (
    merged["text"]
    .fillna("")
    .str.replace(r"<LH>", "", regex=True)
    .str.replace(r"[^a-zA-Z\s#]", "", regex=True)
    .str.lower()
)

# Step 3: Split Data
train_data = merged[merged["identification"] == "train"].dropna(subset=["emotion"])
test_data = merged[merged["identification"] == "test"]

# Step 4: Feature Engineering
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')  # Improved features
X_train = vectorizer.fit_transform(train_data["cleaned_text"])
y_train = train_data["emotion"]
X_test = vectorizer.transform(test_data["cleaned_text"])

# Handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=train_data["emotion"].unique(), 
    y=train_data["emotion"]
)
class_weight_dict = dict(zip(train_data["emotion"].unique(), class_weights))

# Step 5: Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight_dict)
model.fit(X_train, y_train)

# Step 6: Predict Emotions for Test Data
test_data["emotion"] = model.predict(X_test)

# Step 7: Create Submission File
submission = test_data[["tweet_id", "emotion"]].rename(columns={"tweet_id": "id"})
submission_path = "/Users/shan/Desktop/資料探勘/com/submission_improved.csv"
submission.to_csv(submission_path, index=False)

print(f"Improved submission file created at {submission_path}")
