from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json

genai.configure(api_key="AIzaSyBZbPbru3ULNbs6RFX5qxvCcWe6I1C7Ypw")
gemini = genai.GenerativeModel("gemini-2.5-flash")

documents = [
    # Network/Recon
    "Just discovered a new way to use nmap for stealthy network scanning without triggering IDS",
    "masscan is way faster than nmap for large scale port scanning, anyone else using it?",
    "best tools for network reconnaissance without getting detected by firewalls",
    "passive recon is underrated, shodan alone can give you so much about a target",

    # Ransomware
    "new ransomware strain spotted targeting Canadian energy sector companies",
    "LockBit 3.0 updated their encryptor, much harder to decrypt without paying",
    "ransomware group leaked 40GB of data from Ottawa government contractor",
    "double extortion ransomware is now the norm, encrypt AND threaten to leak",

    # Phishing
    "crafted a convincing phishing page that bypasses Microsoft 365 MFA",
    "spear phishing campaigns targeting Canadian banks are increasing this quarter",
    "evilginx2 is still the best tool for credential harvesting via reverse proxy",
    "how to make phishing emails that bypass spam filters",

    # Exploits
    "new zero day in Windows kernel privilege escalation, no patch yet",
    "CVE-2024-1234 proof of concept just dropped on github, patch your systems",
    "zero day market prices have gone up, iOS exploits now worth over 2 million",
    "exploit development for beginners, understanding buffer overflows",

    # Dark web
    "new dark web marketplace launched after previous one got seized by FBI",
    "which markets are still reliable after the recent law enforcement takedowns?",
    "cryptocurrency mixing services getting harder to use after recent crackdowns",
    "vendor ratings on dark web forums are easily manipulated, dont trust them",

    # CTF / Learning
    "just finished HackTheBox machine, learned a lot about privilege escalation",
    "best CTF platforms for learning web application penetration testing",
    "writeup for last weekend CTF challenge, SQL injection to RCE",
    "OSCP certification worth it in 2024, thinking of starting the course",
]

CONTEXT = (
    "I have a dataset of posts scraped from hacker and cybersecurity forums. "
    "The posts discuss topics such as malware, exploits, vulnerabilities, "
    "dark web markets, phishing, and other cyber threats."
)

TEXT_DELIMITER = "####"

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

print(f"Loaded {len(documents)} documents")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=3,
    verbose=True
)

topics, probs = model.fit_transform(documents)

topic_info = model.get_topic_info()
for _, row in topic_info.iterrows():
    if row["Topic"] == -1:
        continue
    rep_docs = row["Representative_Docs"]
    prompt = generate_prompt(rep_docs)
    response = gemini.generate_content(prompt)
    try:
        text = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(text)
        print(f"Topic {row['Topic']}:")
        print(f"  Name: {result['topic_name']}")
        print(f"  Description: {result['topic_description']}")
    except:
        print(f"Topic {row['Topic']}: {response.text.strip()}")