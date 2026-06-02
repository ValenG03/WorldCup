import streamlit as st
import pandas as pd
import plotly.express as px
import tweepy

from datetime import datetime
from transformers import pipeline
from streamlit_autorefresh import st_autorefresh


# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="World Cup Sentiment Tracker",
    layout="wide"
)

st.title("Real-Time World Cup Sentiment Tracker")
st.write("Live X/Twitter sentiment during World Cup matches")


# -------------------------
# API TOKEN
# -------------------------

BEARER_TOKEN = st.secrets["X_BEARER_TOKEN"]


# -------------------------
# LOAD MODEL
# -------------------------

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


sentiment_model = load_model()


# -------------------------
# X CLIENT
# -------------------------

@st.cache_resource
def load_client():
    return tweepy.Client(
        bearer_token=BEARER_TOKEN,
        wait_on_rate_limit=True
    )


client = load_client()


# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.header("Match settings")

query = st.sidebar.text_input(
    "Search query",
    '"World Cup" lang:en -is:retweet'
)

max_tweets = st.sidebar.slider(
    "Tweets per refresh",
    10,
    100,
    30
)

refresh_seconds = st.sidebar.slider(
    "Refresh every seconds",
    10,
    120,
    30
)

goal_label = st.sidebar.text_input(
    "Goal event label",
    "Goal!"
)

add_goal = st.sidebar.button("Mark goal now")


# -------------------------
# SESSION STATE
# -------------------------

if "data" not in st.session_state:
    st.session_state.data = []

if "goals" not in st.session_state:
    st.session_state.goals = []

if add_goal:
    st.session_state.goals.append({
        "time": datetime.now(),
        "label": goal_label
    })


# -------------------------
# AUTO REFRESH
# -------------------------

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="refresh"
)


# -------------------------
# FUNCTIONS
# -------------------------

def analyze_text(text):
    result = sentiment_model(text[:512])[0]
    label = result["label"].lower()
    score = round(result["score"], 3)

    return label, score


def get_tweets():
    response = client.search_recent_tweets(
        query=query,
        max_results=max_tweets,
        tweet_fields=["created_at", "lang"]
    )

    if response.data is None:
        return []

    new_rows = []

    for tweet in response.data:
        label, score = analyze_text(tweet.text)

        new_rows.append({
            "time": datetime.now(),
            "tweet": tweet.text,
            "sentiment": label,
            "score": score
        })

    return new_rows


# -------------------------
# GET LIVE TWEETS
# -------------------------

try:
    new_data = get_tweets()
    st.session_state.data.extend(new_data)

except Exception as e:
    st.error("Error pulling tweets")
    st.write(e)


# -------------------------
# DATAFRAME
# -------------------------

df = pd.DataFrame(st.session_state.data)

if df.empty:
    st.warning("Waiting for tweets...")
    st.stop()


# Keep app light
df = df.tail(500)


# -------------------------
# METRICS
# -------------------------

positive = len(df[df["sentiment"] == "positive"])
neutral = len(df[df["sentiment"] == "neutral"])
negative = len(df[df["sentiment"] == "negative"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total tweets", len(df))
col2.metric("Positive", positive)
col3.metric("Neutral", neutral)
col4.metric("Negative", negative)


# -------------------------
# CHART 1: SENTIMENT COUNT
# -------------------------

count_df = df["sentiment"].value_counts().reset_index()
count_df.columns = ["sentiment", "count"]

fig_count = px.bar(
    count_df,
    x="sentiment",
    y="count",
    title="Current Sentiment Count"
)

st.plotly_chart(fig_count, use_container_width=True)


# -------------------------
# CHART 2: SENTIMENT OVER TIME
# -------------------------

df["minute"] = pd.to_datetime(df["time"]).dt.floor("min")

time_df = (
    df.groupby(["minute", "sentiment"])
    .size()
    .reset_index(name="count")
)

fig_time = px.line(
    time_df,
    x="minute",
    y="count",
    color="sentiment",
    title="Sentiment Shift Over Time",
    markers=True
)

for goal in st.session_state.goals:
    fig_time.add_vline(
        x=goal["time"],
        line_dash="dash",
        annotation_text=goal["label"]
    )

st.plotly_chart(fig_time, use_container_width=True)


# -------------------------
# LATEST TWEETS
# -------------------------

st.subheader("Latest Tweets")

latest = df.tail(10).sort_values("time", ascending=False)

for _, row in latest.iterrows():
    st.write(
        f"**{row['sentiment'].upper()}** "
        f"({row['score']})"
    )
    st.write(row["tweet"])
    st.divider()

    