import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

st.set_page_config(page_title="World Cup Sentiment", layout="wide")

st.title("Qatar 2022 World Cup Sentiment Analysis")
st.write("Tweets dataset + Hugging Face sentiment + Streamlit dashboard")


@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


@st.cache_data
def load_data():
    df = pd.read_csv("tweets.csv")

    text_cols = ["tweet", "Tweet", "text", "Text", "content"]
    date_cols = ["date", "Date", "created_at", "timestamp", "Datetime"]

    text_col = next(c for c in text_cols if c in df.columns)
    date_col = next(c for c in date_cols if c in df.columns)

    df = df[[date_col, text_col]].copy()
    df.columns = ["date", "tweet"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna()
    df = df.sort_values("date")

    return df


def get_sentiment(text):
    result = model(str(text)[:512])[0]
    return result["label"].lower()


model = load_model()
df = load_data()

st.sidebar.header("Settings")

sample_size = st.sidebar.slider(
    "Tweets to analyze",
    100,
    3000,
    500
)

freq = st.sidebar.selectbox(
    "Time interval",
    ["1H", "3H", "6H", "12H", "1D"]
)

df = df.head(sample_size)

if "sentiment" not in df.columns:
    with st.spinner("Analyzing sentiment..."):
        df["sentiment"] = df["tweet"].apply(get_sentiment)


total = len(df)
pos = len(df[df["sentiment"] == "positive"])
neu = len(df[df["sentiment"] == "neutral"])
neg = len(df[df["sentiment"] == "negative"])

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total tweets", total)
c2.metric("Positive", pos)
c3.metric("Neutral", neu)
c4.metric("Negative", neg)


st.subheader("Sentiment Distribution")

count_df = df["sentiment"].value_counts().reset_index()
count_df.columns = ["sentiment", "count"]

fig1 = px.bar(
    count_df,
    x="sentiment",
    y="count",
    title="Positive / Neutral / Negative Tweets"
)

st.plotly_chart(fig1, use_container_width=True)


st.subheader("Opinion Shift During Qatar 2022")

df["period"] = df["date"].dt.floor(freq)

time_df = (
    df.groupby(["period", "sentiment"])
    .size()
    .reset_index(name="count")
)

fig2 = px.line(
    time_df,
    x="period",
    y="count",
    color="sentiment",
    markers=True,
    title="Sentiment Over Time"
)

st.plotly_chart(fig2, use_container_width=True)


st.subheader("Sentiment Share Over Time")

share_df = time_df.copy()
share_df["total"] = share_df.groupby("period")["count"].transform("sum")
share_df["share"] = share_df["count"] / share_df["total"]

fig3 = px.area(
    share_df,
    x="period",
    y="share",
    color="sentiment",
    title="Public Opinion Share Over Time"
)

st.plotly_chart(fig3, use_container_width=True)


st.subheader("Sample Tweets")

st.dataframe(
    df[["date", "tweet", "sentiment"]].tail(20),
    use_container_width=True
)