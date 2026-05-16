from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
from stackapi import StackAPI
import html
import re
import time

CONTEXT = (
    "I have a dataset of posts scraped from hacker and cybersecurity forums. "
    "The posts discuss topics such as malware, exploits, vulnerabilities, "
    "dark web markets, phishing, and other cyber threats."
)

TEXT_DELIMITER = "####"

genai.configure(api_key="")
gemini = genai.GenerativeModel("gemini-2.5-flash")

site = StackAPI('security')
questions = site.fetch('questions', 
    sort='activity',
    order='desc',
    pagesize=100,
    filter='withbody'
)

documents = []
for q in questions['items']:
    title = q.get('title', '')
    body = q.get('body', '')

    body_clean = re.sub(r'<[^>]+>', '', body)
    body_clean = html.unescape(body_clean)

    combined = title + " " + body_clean
    combined = " ".join(combined.split())

    documents.append(combined)


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

print(f"Collected {len(documents)} documents")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=3,
    verbose=True
)

topics, probs = model.fit_transform(documents)
topic_info = model.get_topic_info()
print(topic_info[["Topic", "Count", "Name"]])

for _, row in topic_info.iterrows():
    if row["Topic"] == -1:
        continue
    rep_docs = row["Representative_Docs"]
    prompt = generate_prompt(rep_docs)
    response = gemini.generate_content(prompt)

    time.sleep(15)

    try:
        text = response.text.strip().replace("```json", "").replace("```python", "").replace("```", "")
        result = json.loads(text)
        print(f"Topic {row['Topic']}:")
        print(f"  Name: {result['topic_name']}")
        print(f"  Description: {result['topic_description']}")
    except:
        print(f"Topic {row['Topic']}: {response.text.strip()}")