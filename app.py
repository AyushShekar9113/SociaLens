import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import random
import praw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pytrends.request import TrendReq
import umap  # NEW
import umap
import hdbscan
# ---- PAGE CONFIG ----
st.set_page_config(page_title="SociaLens: AI-Powered Social Media Clustering", layout="wide")
st.title("📊 SociaLens: AI-Powered Social Media Text Clustering")


# ---- Sidebar Mode ----
st.sidebar.header("🔧 Mode")
mode = st.sidebar.radio("Choose Mode",["Clustering", "Trending","Post Analyzer", "🧠 Bias Detector"])

# ---- GOOGLE TRENDS MODE ----
if mode == "Trending":
    st.subheader("🌐 Google Trends Analysis")
    trend_mode = st.radio("Select Trend Type", ["Single Trend", "Compare Trends", "Geo Trends"])

    pytrends = TrendReq(hl='en-US', tz=360)

    if trend_mode == "Single Trend":
        keyword = st.text_input("Enter keyword for trend analysis", value="AI")
        if st.button("Search Trends"):
            pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
            data = pytrends.interest_over_time()
            if not data.empty:
                st.success(f"📈 Showing search trend for: {keyword}")
                fig = px.line(data, x=data.index, y=keyword, title=f"Interest Over Time: {keyword}")
                st.plotly_chart(fig)
            else:
                st.warning("No trend data found.")

    elif trend_mode == "Compare Trends":
        keyword1 = st.text_input("Keyword 1", value="AI")
        keyword2 = st.text_input("Keyword 2", value="Machine Learning")
        keyword3 = st.text_input("Keyword 3", value="Data Science")
        keywords = [kw for kw in [keyword1, keyword2, keyword3] if kw.strip()]
        if st.button("Compare Trends"):
            if len(keywords) < 2:
                st.warning("Please enter at least two keywords to compare.")
            else:
                pytrends.build_payload(keywords, timeframe='today 3-m')
                data = pytrends.interest_over_time()
                if not data.empty:
                    fig = px.line(data, x=data.index, y=keywords, title="Trend Comparison")
                    st.plotly_chart(fig)
                else:
                    st.warning("No data found.")

    elif trend_mode == "Geo Trends":
        keyword = st.text_input("Enter a keyword", value="AI")
        geo_region = st.text_input("Enter Geo Code (e.g. US, IN, GB)", value="IN")
        if st.button("Show Geo Trends"):
            pytrends.build_payload([keyword], timeframe='today 3-m', geo=geo_region)
            region_data = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
            region_data = region_data.sort_values(by=keyword, ascending=False).reset_index()
            if not region_data.empty:
                st.success(f"🌍 Top regions for: {keyword}")
                fig = px.bar(region_data.head(10), x='geoName', y=keyword, title=f"Interest by Region: {keyword}")
                st.plotly_chart(fig)
            else:
                st.warning("No region data found.")

    st.stop()


from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Rename to avoid conflict with Bias Detector tab
def load_emotion_bias_model():
    from transformers import BertTokenizer, BertForSequenceClassification
    tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")
    model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert")
    return tokenizer, model

def classify_emotion_bias(text, tokenizer, model):
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ["Neutral", "Toxic"]
    pred = torch.argmax(probs).item()
    return labels[pred]


def analyze_image(image):
    # Convert PIL to OpenCV format
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # OCR for text
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray)

    # Emotion Detection
    emotion_result = None
    try:
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_result = (dominant_emotion, 100)  # Confidence is not always provided clearly
    except Exception as e:
        print("Emotion analysis failed:", e)

    return extracted_text.strip(), emotion_result

def classify_top_emotions(text, tokenizer, model, top_n=3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    top_probs, top_idxs = torch.topk(probs, k=top_n, dim=1)
    top_probs = top_probs[0].tolist()
    top_idxs = top_idxs[0].tolist()
    
    label_dict = model.config.id2label
    results = [(label_dict[idx], round(prob * 100, 2)) for idx, prob in zip(top_idxs, top_probs)]
    return results

if mode == "🧠 Bias Detector":
    st.subheader("🧠 Bias Detector")
    st.markdown("Detect potential political bias in Reddit posts or text content.")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    @st.cache_resource
    def load_bias_model():
        model_name = "SamLowe/roberta-base-go_emotions"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model

    def classify_bias(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_idx = torch.argmax(probs, dim=1).item()

        
        label_dict = model.config.id2label
        return label_dict[top_idx]

    user_input = st.text_area("Paste Reddit post or any text to check bias:", height=150)
    uploaded_image = st.file_uploader("🖼️ Or upload an image (with text or face)", type=["jpg", "jpeg", "png"])
    analyze_clicked = st.button("🔍 Detect Bias/Emotion")
    if analyze_clicked:
        if user_input.strip():
            with st.spinner("Analyzing..."):
                tokenizer, model = load_bias_model()
                result = classify_bias(user_input, tokenizer, model)
                st.success(f"🔍 Detected Emotion/Bias: **{result}**")
            
        elif uploaded_image:
            with st.spinner("Processing image..."):
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                extracted_text, face_emotion = analyze_image(image)
            if extracted_text:
                st.markdown("**📄 Extracted Text from Image:**")
                st.write(extracted_text)
                tokenizer, model = load_emotion_bias_model()
                bias_result = classify_emotion_bias(extracted_text, tokenizer, model)

                top_emotions = classify_top_emotions(extracted_text, tokenizer, model)

                st.markdown("**🧠 Detected Bias/Emotion from Extracted Text:**")
                for label, prob in top_emotions:
                    st.markdown(f"- {label.capitalize()}: {prob}%")
            if face_emotion:
                st.markdown(f"**😊 Detected Facial Emotion:** `{face_emotion[0].capitalize()}` ({face_emotion[1]}%)")
            if not extracted_text and not face_emotion:
                st.warning("😐 Couldn't find any readable text or detectable face.")
        else:
            st.warning("Please enter some text or image.")

    st.stop()
import streamlit as st
import easyocr
import cv2
def extract_text_from_image(uploaded_file):
    image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image)
    extracted_text = " ".join([res[1] for res in results])
    return extracted_text
import whisper
import tempfile
import ffmpeg

def extract_text_from_video(uploaded_file):
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video.flush()
        result = model.transcribe(temp_video.name)
        return result["text"]

from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')

# Initialize summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_and_extract_claims(text):
    # 1. Summarize
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    # 2. Naive claim extraction: break into sentences (improve later with LLM)
    claims = sent_tokenize(text)
    return summary, claims

import requests

def get_groq_verdict(claim):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer (GROQAPI_KEY)",
        "Content-Type": "application/json"
    }

    prompt = f"""You are a fact-checking assistant. Analyze the following claim and classify it as "True", "False", or "Partial". Provide a short explanation. Respond only in this JSON format:
{{
  "verdict": "...",
  "explanation": "..."
}}

Claim: {claim}
"""

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful AI fact-checking assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Try to parse verdict and explanation from response
        import json
        result = json.loads(content)
        return result.get("verdict", "❓"), result.get("explanation", "")
    except Exception as e:
        return "⚠️ Could not parse", str(e)

import requests

def fact_check_claim(claim, serper_api_key, groq_api_key):
    # Step 1: Search the web for evidence using Serper.dev
    serper_url = "https://google.serper.dev/search"
    serper_headers = {
        "X-API-KEY": "684af957cf7023b1e79943df3d4718d46be55c8d",
        "Content-Type": "application/json"
    }
    serper_payload = {
        "q": claim,
        "num": 5
    }
    serper_response = requests.post(serper_url, headers=serper_headers, json=serper_payload)

    if serper_response.status_code != 200:
        return {
            "verdict": "⚠️ Could not parse",
            "explanation": f"Serper Error: {serper_response.text}",
            "evidence_snippets": [],
            "sources": []
        }

    serper_results = serper_response.json().get("organic", [])
    evidence_snippets = []
    sources = []

    for result in serper_results:
        snippet = result.get("snippet") or result.get("description") or ""
        title = result.get("title", "")
        link = result.get("link", "")
        evidence_snippets.append(snippet)
        sources.append((title, link))

    combined_evidence = "\n\n".join(evidence_snippets[:3])  # Just the top 3 for Groq input

    # Step 2: Ask Groq LLM for a final verdict
    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    groq_headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }
    groq_payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You're an expert fact-checking assistant. Given a factual claim and some online evidence, provide a verdict (True, False, Partial) and a brief explanation."
            },
            {
                "role": "user",
                "content": f"Claim: {claim}\n\nEvidence:\n{combined_evidence}\n\nReturn answer as:\nVerdict: ✅ True / ❌ False / ⚠️ Partial\nExplanation: ..."
            }
        ],
        "temperature": 0.3
    }

    groq_response = requests.post(groq_url, headers=groq_headers, json=groq_payload)

    if groq_response.status_code != 200:
        return {
            "verdict": "⚠️ Could not parse",
            "explanation": f"Groq Error: {groq_response.text}",
            "evidence_snippets": evidence_snippets,
            "sources": sources
        }

    groq_answer = groq_response.json()["choices"][0]["message"]["content"]

    # Extracting verdict & explanation from Groq's reply
    verdict_line = next((line for line in groq_answer.splitlines() if "Verdict" in line), "Verdict: ⚠️ Could not extract")
    explanation_lines = [line for line in groq_answer.splitlines() if "Explanation" in line or not line.startswith("Verdict")]
    explanation_text = "\n".join(explanation_lines).strip()

    return {
        "verdict": verdict_line,
        "explanation": explanation_text,
        "evidence_snippets": evidence_snippets,
        "sources": sources
    }

extracted_text = ""
if mode =="Post Analyzer":  
    st.header("📊 Post Analyzer - Fact Check & Summary")
    uploaded_file = st.file_uploader("Upload a post (image/video/text)", type=["png", "jpg", "jpeg", "mp4", "txt"])

    text_input = st.text_area("Or paste post content here", placeholder="Paste any content you want to analyze...")

    analyze_btn = st.button("🧠 Analyze Post")

    if analyze_btn:
        if uploaded_file:
            file_type = uploaded_file.type

        if file_type.startswith("image/"):
            extracted_text = extract_text_from_image(uploaded_file)
        elif file_type.startswith("video/"):
            extracted_text = extract_text_from_video(uploaded_file)
        elif file_type == "text/plain":
            extracted_text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file type.")

    elif text_input.strip():
        extracted_text = text_input.strip()

    if extracted_text:
        with st.spinner("Summarizing and extracting factual claims..."):
            summary, claims = summarize_and_extract_claims(extracted_text)
        st.session_state["summary"] = summary
        st.session_state["claims"] = claims
        st.session_state["post_input"] = extracted_text

        st.rerun()
    if "summary" in st.session_state and "claims" in st.session_state:
        st.subheader("📝 Summary")
        st.success(st.session_state["summary"])
    
        st.subheader("📌 Extracted Claims")
        st.subheader("🔍 Fact Check Results")

        from difflib import SequenceMatcher

        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()
        from duckduckgo_search import DDGS
        import requests

        def web_search(query, num_results=5):
            headers = {
                "X-API-KEY": "684af957cf7023b1e79943df3d4718d46be55c8d",  # Your key
                "Content-Type": "application/json"
            }
            data = {
                "q": query,
                "num": num_results
            }
            response = requests.post("https://google.serper.dev/search", headers=headers, json=data)
            if response.status_code == 200:
                results = response.json()
                return results.get("organic", [])
            else:
                return []


        for idx, claim in enumerate(st.session_state["claims"], 1):
            with st.spinner(f"🔍 Fact-checking Claim {idx}: {claim}"):
                result = fact_check_claim(
                    claim,
                    serper_api_key="your_serper_api_key_here",
                    groq_api_key="your_groq_api_key_here"
                )

                st.markdown(f"### Claim {idx}: {claim}")
                st.markdown(f"**🧠 Verdict:** {result['verdict']}")
                st.markdown(f"**💬 Explanation:** {result['explanation']}")

                def toggle_evidence(key):
                    st.session_state[key] = not st.session_state.get(key, False)

                toggle_key = f"show_more_evidence_{idx}"

# Use a different key for the button (e.g. prefix with "btn_")
                st.button("🔎 More Evidence", key=f"btn_{toggle_key}", on_click=toggle_evidence, args=(toggle_key,))

                if st.session_state.get(toggle_key, False):
                    if result["evidence_snippets"]:
                        st.markdown("**🧾 More Evidence Snippets:**")
                        for snippet in result["evidence_snippets"]:
                            st.markdown(f"- {snippet[:300]}...")

                    if result["sources"]:
                        st.markdown("**🔗 Additional Sources:**")
                        for title, link in result["sources"]:
                            st.markdown(f"- [{title}]({link})")
                    else:
            # Show only 1–2 top evidence snippets by default
                        if result["evidence_snippets"]:
                            st.markdown("**🧾 Evidence Snippet:**")
                            st.markdown(f"- {result['evidence_snippets'][0][:300]}...")

                        if result["sources"]:
                            st.markdown("**🔗 Top Source:**")
                            st.markdown(f"- [{result['sources'][0][0]}]({result['sources'][0][1]})")

                    st.markdown("---")

        # Use Groq API for final verdict
            #     groq_output = get_groq_verdict(claim, evidence_snippets[:5])  # Limit to 5 snippets for prompt length

            #     try:
            #         import json
            #         parsed = json.loads(groq_output)
            #         raw_verdict = parsed.get("verdict", "Unknown").strip().lower()
            #         explanation = parsed.get("explanation", "No explanation given.")

            # # Convert raw verdict to emoji-based format
            #         if raw_verdict == "true":
            #             verdict = "✅ True"
            #         elif raw_verdict == "false":
            #             verdict = "❌ False"
            #         elif raw_verdict == "partial":
            #             verdict = "❓ Partial"
            #         else:
            #             verdict = f"⚠️ Unknown ({raw_verdict})"

            #     except Exception as e:
            #         verdict = "⚠️ Could not parse"
            #         explanation = str(groq_output)
        # STEP 2: Cross-check with Groq
        # Display


    if "post_input" in st.session_state:
        st.markdown("### 📝 Post Summary & Fact Check (Preview)")
        st.write(st.session_state["post_input"])
        st.info("⏳ Running summarization and fact-checking pipeline...")
        # Next steps: NLP processing
    st.stop()

# ---- Sidebar Inputs ----
st.sidebar.header("🔧 Settings")
post_limit = st.sidebar.slider("Number of Posts to Fetch", 10, 200, 50)
n_clusters = st.sidebar.slider("KMeans: Number of Clusters", 2, 10, 4)

# ---- Reddit API Setup ----
import praw

reddit = praw.Reddit(
    client_id="cT3BWSAIMCxyiNM3YOB6BA",
    client_secret="By1gye63JdaPHJSR_r9aDL1VHOGEvg",
    user_agent="socialens by u/Nearby_Mud_8458"
)

reddit.read_only = True
import feedparser
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Optional: for detecting sentiment
classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment", top_k=1)

def summarize_article(url, max_chars=500):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        content = " ".join(p.get_text() for p in paragraphs[:5])
        return content.strip()[:max_chars] or "Summary not available."
    except:
        return "Summary not available."
from googletrans import Translator
from textblob import TextBlob
import feedparser
import re

# Optional: Plug in a HuggingFace summarizer (optional, or use entry.summary directly)

translator = Translator()


import re
import feedparser
from textblob import TextBlob
from deep_translator import GoogleTranslator

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def fetch_news_by_location(country, city=None, n=10, show_sentiment=True, translate_to="English"):
    country_code_map = {
        "India": "IN",
        "United States": "US",
        "United Kingdom": "GB",
        "Germany": "DE",
        "France": "FR",
        "Canada": "CA",
        "Australia": "AU"
    }

    language_map = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
        "Tamil": "ta",
        "Telugu": "te",
        "Marathi": "mr"
    }

    country_code = country_code_map.get(country, "US")
    lang_code = language_map.get(translate_to, "en")

    query = f"{city} {country}" if city else country
    query = query.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en&gl={country_code}&ceid={country_code}:en"

    feed = feedparser.parse(rss_url)
    headlines = []

    for entry in feed.entries[:n]:
        title = entry.title
        link = entry.link
        summary = clean_html(entry.get("summary", ""))
        short_summary = summary[:200] + "..." if len(summary) > 200 else summary

        # Translate using deep-translator
        try:
            if lang_code != "en":
                translated_summary = GoogleTranslator(source='auto', target=lang_code).translate(short_summary)
            else:
                translated_summary = short_summary
        except Exception as e:
            translated_summary = short_summary + " (translation failed)"

        sentiment = analyze_sentiment(short_summary) if show_sentiment else None

        headlines.append({
            "title": title,
            "link": link,
            "summary": translated_summary,
            "sentiment": sentiment
        })

    return headlines

# === NEWS FETCH FUNCTION ===
# def fetch_news_by_location(country, city=None, n=20):
#     country_code_map = {
#         "India": "IN",
#         "United States": "US",
#         "United Kingdom": "GB",
#         "Germany": "DE",
#         "France": "FR",
#         "Canada": "CA",
#         "Australia": "AU"
#     }
#     country_code = country_code_map.get(country, "US")
#     query = f"{city} {country}" if city else country
#     query = query.replace(" ", "+")
#     rss_url = f"https://news.google.com/rss/search?q={query}&hl=en&gl={country_code}&ceid={country_code}:en"

#     headlines = []
#     feed = feedparser.parse(rss_url)
#     for entry in feed.entries[:n]:
#         title = entry.title
#         summary = entry.get("summary", "")
#         headlines.append(f"📰 {title}\n\n{summary}\n")
#     return headlines

# --------------------------------------------
# 🔁 Tweepy Auth Setup (Put this at the top)
# --------------------------------------------
import tweepy
import streamlit as st

# Replace with your actual credentials
api_key = "WfTdrwZcvjYq7hMBvAk7rCF1N"
api_secret = "38M456bYOxR1h01df8EX35BMusBv98Jo5gxVjUkObwKUooqF9W"
access_token = "1928715040263139328-YKxNFLYdwtgUACIjU401KceNgMgVyp"
access_token_secret = "cNSxRrOD0FcoQj04XmnwLTzkGMBzldef9z8fSVT5d3E5V"

auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# --------------------------------------------
# 🌍 Function: Get Country-Wise Trends
# --------------------------------------------
import requests
from bs4 import BeautifulSoup



import requests
from bs4 import BeautifulSoup

def get_trends_from_trends24(country, max_trends=10):
    """
    Scrapes top Twitter trends from trends24.in for a given country or city page.

    Args:
        country (str): Location name (must match slug on trends24.in).
        max_trends (int): Max number of trends to return.

    Returns:
        dict: {'<country>': [trend1, trend2, ...]}
    """
    country_slug_map = {
        "India": "india",
        "United States": "united-states",
        "United Kingdom": "united-kingdom",
        "Japan": "japan",
        "Canada": "canada",
        "Australia": "australia"
    }

    slug = country_slug_map.get(country.lower().title(), country.lower().replace(" ", "-"))
    url = f"https://trends24.in/{slug}/"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Try trend-card first (most reliable)
        trend_card = soup.find("div", class_="trend-card")
        if trend_card:
            trend_list = trend_card.find("ol", class_="trend-card__list")
            trends = [a.get_text(strip=True) for a in trend_list.find_all("a")]
            return {country: trends[:max_trends]}

        # Fallback: find all <ol> with trends manually
        all_trends = []
        for ol in soup.find_all("ol", class_="trend-card__list"):
            all_trends.extend([a.get_text(strip=True) for a in ol.find_all("a")])

        return {country: all_trends[:max_trends]} if all_trends else {country: []}

    except Exception as e:
        print(f"⚠️ Error fetching trends for {country}: {e}")
        return {country: []}


def fetch_twitter_posts(query="AI", max_tweets=100):
    # Securely insert your credentials
    api_key = "WfTdrwZcvjYq7hMBvAk7rCF1N"
    api_secret = "38M456bYOxR1h01df8EX35BMusBv98Jo5gxVjUkObwKUooqF9W"
    bearer_token = "AAAAAAAAAAAAAAAAAAAAACqM2AEAAAAAJFLFSm6b2AJLCOaVPjLNdxPVVn0%3DUQe8eAk4tyWAkOocs9SpxuGfHfG1q1xEVeqmLldPoG5n1stFc5"

    client = tweepy.Client(bearer_token=bearer_token)

    tweets = []
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_tweets, 100),  # API allows max 100 per call
            tweet_fields=["text"]
        )

        if response.data:
            for tweet in response.data:
                tweets.append(tweet.text)
    except Exception as e:
        print(f"Twitter API Error: {e}")
        return []

    return tweets


# ---- REDDIT DATA FETCH FUNCTION (with praw) ----
def fetch_reddit_posts(subreddit_name="all", sort_by="hot", timeframe="day", limit=100):
    posts = []
    try:
        subreddit = reddit.subreddit(subreddit_name)

        if sort_by == "hot":
            submissions = subreddit.hot(limit=limit)
        elif sort_by == "new":
            submissions = subreddit.new(limit=limit)
        elif sort_by == "top":
            submissions = subreddit.top(time_filter=timeframe, limit=limit)
        else:
            return []

        for submission in submissions:
            if not submission.stickied:
                posts.append(submission.title + " " + submission.selftext)

        return posts
    except Exception as e:
        st.error(f"Reddit Fetch Error: {e}")
        return []

# ---- STEP 1: DATA SOURCE SELECTION ----
st.sidebar.header("Step 1: Select Data Source")
data_source = st.sidebar.radio("📡 Choose data source", ["Sample (Offline)", "Reddit (Live)", "News (Live)","Twitter (Live)"], key="data_source_radio")
n_samples = st.sidebar.slider("Number of Posts", 50, 1000, 200, key="n_samples_slider")

random_texts = []

if data_source == "Reddit (Live)":
    subreddit_name = st.sidebar.text_input("Enter Subreddit", "AskReddit", key="subreddit_input")
    sort_by = st.sidebar.selectbox("Sort by", ["hot", "new", "top"], key="sort_by_select")
    timeframe = st.sidebar.selectbox("Timeframe (if top)", ["day", "week", "month", "year"], key="timeframe_select")

    if st.sidebar.button("Fetch Reddit Data", key="fetch_button"):
        with st.spinner("🔄 Fetching Reddit posts..."):
            random_texts = fetch_reddit_posts(subreddit_name, sort_by, timeframe, n_samples)
        if random_texts:
            st.success(f"✅ Fetched {len(random_texts)} posts from r/{subreddit_name}")
        else:
            st.warning("⚠️ No posts found. Try a different subreddit or reduce sample size.")

elif data_source == "News (Live)":
    st.sidebar.markdown("### 🌍 Location-Based News")

    # Country selector
    selected_country = st.sidebar.selectbox("Select Country", [
        "India", "United States", "United Kingdom", "Germany", "France", "Canada", "Australia"
    ])

    # Optional city input
    selected_city = st.sidebar.text_input("Enter City (optional)", key="city_input")

    # Number of articles
    n_samples = st.sidebar.slider("Number of articles", 5, 25, 10)

    # Optional: sentiment checkbox
    show_sentiment = st.sidebar.checkbox("Show Sentiment", value=True)
    translate_to = st.sidebar.selectbox("Translate Summary To", [
        "English", "Hindi", "Kannada", "Tamil", "Telugu", "Marathi"
    ])
    # Fetch button
    if st.sidebar.button("Fetch News", key="news_button"):
        st.info("🔄 Fetching location-based news...")
        news_items = fetch_news_by_location(selected_country, selected_city, n=n_samples, show_sentiment=show_sentiment,translate_to=translate_to)

        if news_items:
            st.success(f"✅ Fetched {len(news_items)} news articles from {selected_city or selected_country}")
            for idx, article in enumerate(news_items, 1):
                st.markdown(f"### {idx}. 📰 [{article['title']}]({article['link']})")
                st.markdown(f"**Summary:** {article['summary']}")
                if show_sentiment and article.get("sentiment"):
                    st.markdown(f"🧠 Sentiment: `{article['sentiment']}`")
                st.markdown("---")
        else:
            st.warning("⚠️ No news found for the specified location.")

elif data_source == "Twitter (Live)":
    twitter_mode = st.sidebar.radio("Choose Twitter Mode", ["Keyword Search", "Geo Trends"])

    if twitter_mode == "Keyword Search":
        twitter_query = st.sidebar.selectbox("Choose Twitter Topic", ["AI", "ChatGPT", "SpaceX", "Climate Change", "Tech News"])
        if st.sidebar.button("Fetch Tweets"):
            random_texts = fetch_twitter_posts(twitter_query, n_samples)
            if random_texts:
                st.success(f"✅ Fetched {len(random_texts)} tweets for '{twitter_query}'")
            else:
                st.error("❌ No tweets fetched. Try a different query.")

    elif twitter_mode == "Geo Trends":
        country = st.sidebar.selectbox("Select Country", ["india", "United States", "Japan", "Canada", "United Kingdom", "Australia"])
        cities_to_fetch = st.sidebar.slider("Number of Cities", 1, 10, 3)
        trends_per_city = st.sidebar.slider("Trends per City", 5, 20, 10)

        if st.sidebar.button("Fetch Trends"):
            with st.spinner("Fetching Twitter geo trends..."):
                trend_dict = get_trends_from_trends24(country=country, max_trends=trends_per_city)


            random_texts = []
            for city, trends in trend_dict.items():
                st.subheader(f"📍 {city}")
                for trend in trends:
                    st.markdown(f"- {trend}")
                    random_texts.append(trend)

            if random_texts:
                st.success(f"✅ Fetched {len(random_texts)} trends from {country}")
            else:
                st.error("❌ Could not fetch any trends.")

else:
    sample_texts = [
        "AI is changing the world! 🚀", "Love this new tech update! 🔥",
    ]
    random_texts = sample_texts

# ---- STEP 2: CONVERT TO DATAFRAME ----
if random_texts:
    df = pd.DataFrame(random_texts, columns=["text"])
    st.write("📂 **Social Media Posts Used:**", df.head())
else:
    st.warning("⚠️ Please load or generate data to proceed.")
    st.stop()

# ---- STEP 3: VECTORIZE TEXT ----
st.subheader("Vectorizing Posts")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text'])

# ---- STEP 4: DIMENSIONALITY REDUCTION WITH UMAP ----
st.subheader("Reducing Dimensions with UMAP")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
try:
    if X.shape[0] == 0 or X.shape[1] == 0:
        st.warning("No vectorized text available for UMAP. Skipping projection.")
    else:
        X_umap = umap_model.fit_transform(X.toarray())
except Exception as e:
    st.error(f"UMAP Error: {e}")
    st.stop()

# ---- STEP 5: CLUSTERING WITH HDBSCAN ----
st.subheader("Clustering with HDBSCAN")
hdb = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
cluster_labels = hdb.fit_predict(X_umap)

df['cluster'] = cluster_labels
df['umap_x'] = X_umap[:, 0]
df['umap_y'] = X_umap[:, 1]

# ---- STEP 6: VISUALIZE RESULTS ----
st.subheader("Reddit Post Clusters (UMAP + HDBSCAN)")
fig = px.scatter(
    df, x='umap_x', y='umap_y',
    color=df['cluster'].astype(str),
    hover_data=['text'],
    title="UMAP + HDBSCAN Clusters"
)
st.plotly_chart(fig, use_container_width=True)

# Optional: Show Cluster Table
with st.expander("📄 View Clustered Data"):
    st.dataframe(df[['text', 'cluster']])
