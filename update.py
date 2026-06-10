from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import json
from stackapi import StackAPI
import html
import re
import time
from datetime import datetime, timezone
import joblib

CONTEXT = (
    "I have a dataset of posts scraped from hacker and cybersecurity forums. "
    "The posts discuss topics such as malware, exploits, vulnerabilities, "
    "dark web markets, phishing, and other cyber threats."
)

TEXT_DELIMITER = "####"

genai.configure(api_key="")
gemini = genai.GenerativeModel("gemini-2.5-flash")

old_documents = joblib.load("documents.joblib")
old_timestamps = joblib.load("timestamps.joblib")
old_ids = joblib.load("question_ids.joblib")
old_centroids = joblib.load("topic_centroids.joblib")
old_labels = joblib.load("topic_labels.joblib")
old_ids_set = set(old_ids)

print(f"Loaded {len(old_documents)} existing documents")
print(f"Loaded {len(old_labels)} existing topic labels")

site = StackAPI('security')
all_questions = []
page = 1

while len(all_questions) < 100:
    questions = site.fetch('questions',
        sort='creation',
        order='desc',
        pagesize=100,
        page=page,
        filter='withbody'
    )
    items = questions.get('items', [])
    if not items:
        break
    all_questions.extend(items)
    page += 1
    time.sleep(3)

new_documents = []
new_timestamps = []
new_ids = []

for q in all_questions:
    question_id = q["question_id"]
    if question_id in old_ids_set:
        continue

    title = q.get('title', '')
    body = q.get('body', '')
    body_clean = re.sub(r'<[^>]+>', '', body)
    body_clean = html.unescape(body_clean)
    combined = title + " " + body_clean
    combined = " ".join(combined.split())

    ts = q.get('creation_date', 0)
    if ts == 0:
        continue

    new_ids.append(question_id)
    new_documents.append(combined)
    new_timestamps.append(datetime.fromtimestamp(ts, tz=timezone.utc))

print(f"Found {len(new_documents)} new posts")

if len(new_documents) == 0:
    print("No new questions found.")
    quit()


all_documents = old_documents + new_documents
all_timestamps = old_timestamps + new_timestamps
all_ids = old_ids + new_ids

# --- RETRAIN BERTOPIC ON ALL DATA ---
# Full retrain on accumulated data — more stable than partial_fit
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer_model = CountVectorizer(stop_words="english")

model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=15,
    verbose=True
)

topics, probs = model.fit_transform(all_documents)
topic_info = model.get_topic_info()
print(f"Retrained — found {len(topic_info)} topics")

# --- COMPUTE NEW CENTROIDS ---
# After retraining, compute centroids for all new topics
def compute_centroids(model, embedding_model):
    centroids = {}
    topic_info = model.get_topic_info()
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        rep_docs = row["Representative_Docs"]
        embeddings = embedding_model.encode(rep_docs)
        centroid = np.mean(embeddings, axis=0)
        centroids[row["Topic"]] = centroid
    return centroids

new_centroids = compute_centroids(model, embedding_model)

SIMILARITY_THRESHOLD = 0.85

new_labels = {}

for new_topic_id, new_centroid in new_centroids.items():
    best_score = 0
    best_old_id = None

    for old_topic_id, old_centroid in old_centroids.items():
        score = cosine_similarity(
            new_centroid.reshape(1, -1),
            old_centroid.reshape(1, -1)
        )[0][0]

        if score > best_score:
            best_score = score
            best_old_id = old_topic_id

    if best_score >= SIMILARITY_THRESHOLD and best_old_id in old_labels:
        new_labels[new_topic_id] = old_labels[best_old_id]
        print(f"Topic {new_topic_id} matched to existing: {old_labels[best_old_id]['topic_name']} (score {best_score:.2f})")
    else:
        print(f"Topic {new_topic_id} is new (best score {best_score:.2f}) — labelling with Gemini...")
        row = topic_info[topic_info["Topic"] == new_topic_id].iloc[0]
        rep_docs = row["Representative_Docs"]
        docs_text = TEXT_DELIMITER.join(rep_docs)
        prompt = f"""
        {CONTEXT}

        Below is a representative set of forum posts delimited with {TEXT_DELIMITER}.

        Please identify the single main topic mentioned in these posts. Return a topic 
        name and topic description. The topic name should be short but descriptive.
        The topic description should not be a complete sentence.

        Return the topic name and description as a python dictionary like this:
        {{"topic_name": "<topicName>", "topic_description": "<topicDescription>"}}

        If you cannot find a good topic label, just say, "No topic identified".

        Forum posts:
        {docs_text}
        """
        try:
            response = gemini.generate_content(prompt)
            time.sleep(20)
            text = response.text.strip().replace("```json", "").replace("```python", "").replace("```", "")
            result = json.loads(text)
            new_labels[new_topic_id] = result
            print(f"  Name: {result['topic_name']}")
            print(f"  Description: {result['topic_description']}")
        except Exception as e:
            print(f"  Gemini failed: {e}")
            new_labels[new_topic_id] = {"topic_name": "Unknown", "topic_description": "Could not label"}

joblib.dump(model, "bertopic_model.joblib")
joblib.dump(all_documents, "documents.joblib")
joblib.dump(all_timestamps, "timestamps.joblib")
joblib.dump(all_ids, "question_ids.joblib")
joblib.dump(new_centroids, "topic_centroids.joblib")
joblib.dump(new_labels, "topic_labels.joblib")

print(f"Saved — model now has {len(all_documents)} total documents")

topics_over_time = model.topics_over_time(all_documents, all_timestamps, nr_bins=5)
print(topics_over_time.to_string())