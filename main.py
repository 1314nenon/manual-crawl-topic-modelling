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

site = StackAPI('security')

all_questions = []
page = 1

while len(all_questions) < 1000:
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

documents = []
timestamps = []
question_ids = []

for q in all_questions:


    title = q.get('title', '')
    body = q.get('body', '')
    body_clean = re.sub(r'<[^>]+>', '', body)
    body_clean = html.unescape(body_clean)

    combined = title + " " + body_clean
    combined = " ".join(combined.split())
    
    ts = q.get('creation_date', 0)
    if ts == 0:
        continue

    question_ids.append(q["question_id"])
    timestamps.append(datetime.fromtimestamp(ts, tz=timezone.utc))
    documents.append(combined)

print(f"Collected {len(documents)} documents")


def generate_prompt(rep_docs):
    docs_text = TEXT_DELIMITER.join(rep_docs)
    
    prompt = f"""
    {CONTEXT}

    Below is a representative set of forum posts delimited with {TEXT_DELIMITER}.

    Please identify the single main topic mentioned in these posts. Return a topic 
    name and topic description. The topic name should be short but descriptive.
    The topic description should not be a complete sentence. A good topic 
    description looks like this:
    "Techniques used to bypass authentication and steal credentials"

    Return the topic name and description as a python dictionary like this:
    {{"topic_name": "<topicName>", "topic_description": "<topicDescription>"}}

    If you cannot find a good topic label, just say, "No topic identified".

    Forum posts:
    {docs_text}
    """
    return prompt

def compute_centroids(model, embedding_model):
    # For each topic, compute the centroid by averaging the embeddings of its representative documents
    centroids = {}
    topic_info = model.get_topic_info()
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        rep_docs = row["Representative_Docs"]
        # Embed the representative docs and average them
        embeddings = embedding_model.encode(rep_docs)
        centroid = np.mean(embeddings, axis=0)
        centroids[row["Topic"]] = centroid
    return centroids

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer_model = CountVectorizer(stop_words="english")

model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=15,
    verbose=True
)

topics, probs = model.fit_transform(documents)
print("Model trained")

topic_info = model.get_topic_info()

centroids = compute_centroids(model, embedding_model)

topic_labels = {}

for _, row in topic_info.iterrows():
    if row["Topic"] == -1:
        continue
    rep_docs = row["Representative_Docs"]
    prompt = generate_prompt(rep_docs)
    response = gemini.generate_content(prompt)
    time.sleep(20)
    try:
        text = response.text.strip().replace("```json", "").replace("```python", "").replace("```", "")
        result = json.loads(text)
        topic_labels[row["Topic"]] = result
        print(f"Topic {row['Topic']}:")
        print(f"  Name: {result['topic_name']}")
        print(f"  Description: {result['topic_description']}")
    except:
        print(f"Topic {row['Topic']}: {response.text.strip()}")

joblib.dump(model, "bertopic_model.joblib")
joblib.dump(documents, "documents.joblib")
joblib.dump(timestamps, "timestamps.joblib")
joblib.dump(question_ids, "question_ids.joblib")
joblib.dump(centroids, "topic_centroids.joblib")
joblib.dump(topic_labels, "topic_labels.joblib")
print("Model saved")

topics_over_time = model.topics_over_time(documents, timestamps, nr_bins=5)
print(topics_over_time.to_string())
