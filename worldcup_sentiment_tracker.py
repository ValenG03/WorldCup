import os
import re
import time
import threading
from collections import deque
from datetime import datetime, timezone

import tweepy
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

if not X_BEARER_TOKEN:
    raise ValueError("Missing X_BEARER_TOKEN in .env file")

MATCH_NAME = "Argentina vs France"

# Change these according to the match you are tracking.
STREAM_RULES = [
    '"Argentina" "France" lang:en -is:retweet',
    '"ARGFRA" lang:en -is:retweet',
    '"World Cup" "Argentina" lang:en -is:retweet',
    '"Messi" "World Cup" lang:en -is:retweet',
    '#FIFAWorldCup lang:en -is:retweet',
]

# Optional: manually record goal moments.
# You can add timestamps during the match.
# Example:
# GOAL_EVENTS = [
#     {"team": "Argentina", "minute": 23, "timestamp": "2026-06-01T18:30:00+00:00"},
# ]
GOAL_EVENTS = []


# Maximum number of analyzed tweets stored in memory.
MAX_TWEETS = 3000

tweet_buffer = deque(maxlen=MAX_TWEETS)

lock = threading.Lock()


# ============================================================
# SENTIMENT MODEL
# ============================================================

# Good model for Twitter/X-like text.
# Labels are usually: negative, neutral, positive.
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_tweet(text: str) -> str:
    """
    Basic cleaning for tweets.
    Keeps the text readable for sentiment analysis.
    """
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Returns sentiment label and confidence score.
    """
    cleaned = clean_tweet(text)

    if not cleaned:
        return "neutral", 0.0

    result = sentiment_model(cleaned[:512])[0]

    label = result["label"].lower()
    score = float(result["score"])

    if label not in ["positive", "negative", "neutral"]:
        # Fallback in case model uses different label format.
        if "pos" in label:
            label = "positive"
        elif "neg" in label:
            label = "negative"
        else:
            label = "neutral"

    return label, score


# ============================================================
# X STREAMING CLIENT
# ============================================================

class WorldCupStream(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        """
        Called whenever a tweet matching the stream rules arrives.
        """
        try:
            text = tweet.text
            sentiment, confidence = analyze_sentiment(text)

            item = {
                "timestamp": datetime.now(timezone.utc),
                "tweet_id": tweet.id,
                "text": text,
                "clean_text": clean_tweet(text),
                "sentiment": sentiment,
                "confidence": confidence,
            }

            with lock:
                tweet_buffer.append(item)

            print(
                f"[{item['timestamp']}] "
                f"{sentiment.upper()} "
                f"({confidence:.2f}) - {item['clean_text'][:100]}"
            )

        except Exception as e:
            print(f"Error processing tweet: {e}")

    def on_errors(self, errors):
        print("Stream errors:", errors)

    def on_connection_error(self):
        print("Connection error. Disconnecting stream.")
        self.disconnect()

    def on_exception(self, exception):
        print("Stream exception:", exception)
        time.sleep(5)


def reset_stream_rules(stream_client: WorldCupStream):
    """
    Deletes previous rules and adds the current match rules.
    """
    existing_rules = stream_client.get_rules()

    if existing_rules.data:
        rule_ids = [rule.id for rule in existing_rules.data]
        stream_client.delete_rules(rule_ids)

    new_rules = []

    for rule in STREAM_RULES:
        new_rules.append(tweepy.StreamRule(value=rule))

    stream_client.add_rules(new_rules)

    print("Active stream rules:")
    active_rules = stream_client.get_rules()

    if active_rules.data:
        for rule in active_rules.data:
            print("-", rule.value)


def start_stream():
    """
    Starts the X filtered stream in a separate thread.
    """
    stream_client = WorldCupStream(
        bearer_token=X_BEARER_TOKEN,
        wait_on_rate_limit=True,
    )

    reset_stream_rules(stream_client)

    print("Starting live X stream...")

    stream_client.filter(
        tweet_fields=["created_at", "lang", "public_metrics"],
        threaded=False,
    )


# ============================================================
# DATA HELPERS
# ============================================================

def get_dataframe() -> pd.DataFrame:
    """
    Converts the in-memory tweet buffer into a pandas DataFrame.
    """
    with lock:
        data = list(tweet_buffer)

    if not data:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "tweet_id",
                "text",
                "clean_text",
                "sentiment",
                "confidence",
            ]
        )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def make_sentiment_timeseries(df: pd.DataFrame, frequency: str = "1min") -> pd.DataFrame:
    """
    Groups tweets by time and sentiment.
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "sentiment", "count"])

    grouped = (
        df.set_index("timestamp")
        .groupby("sentiment")
        .resample(frequency)
        .size()
        .reset_index(name="count")
    )

    return grouped


def make_sentiment_share(df: pd.DataFrame, frequency: str = "1min") -> pd.DataFrame:
    """
    Calculates sentiment share over time.
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "sentiment", "count", "share"])

    grouped = make_sentiment_timeseries(df, frequency)

    total_by_time = grouped.groupby("timestamp")["count"].transform("sum")
    grouped["share"] = grouped["count"] / total_by_time

    return grouped


def get_latest_tweets(df: pd.DataFrame, n: int = 10) -> list[html.Div]:
    """
    Creates HTML cards for latest tweets.
    """
    if df.empty:
        return [html.Div("No tweets yet.")]

    latest = df.sort_values("timestamp", ascending=False).head(n)

    cards = []

    for _, row in latest.iterrows():
        sentiment = row["sentiment"]
        confidence = row["confidence"]
        text = row["clean_text"]

        cards.append(
            html.Div(
                [
                    html.Div(
                        f"{sentiment.upper()} · confidence {confidence:.2f}",
                        style={
                            "fontWeight": "bold",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(text),
                    html.Div(
                        str(row["timestamp"]),
                        style={
                            "fontSize": "11px",
                            "color": "#666",
                            "marginTop": "4px",
                        },
                    ),
                ],
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "marginBottom": "8px",
                    "backgroundColor": "#fafafa",
                },
            )
        )

    return cards


def add_goal_markers(fig):
    """
    Adds vertical lines to a Plotly figure for goals.
    """
    for goal in GOAL_EVENTS:
        ts = pd.to_datetime(goal["timestamp"])
        label = f"Goal: {goal['team']} {goal['minute']}'"

        fig.add_vline(
            x=ts,
            line_dash="dash",
            annotation_text=label,
            annotation_position="top left",
        )

    return fig


# ============================================================
# DASH APP
# ============================================================

app = Dash(__name__)

app.title = "World Cup Sentiment Tracker"

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "20px",
        "backgroundColor": "#f4f6f8",
    },
    children=[
        html.H1("Real-Time World Cup Sentiment Tracker"),
        html.H3(MATCH_NAME),

        html.Div(
            id="summary-cards",
            style={
                "display": "flex",
                "gap": "12px",
                "marginBottom": "20px",
            },
        ),

        dcc.Graph(id="sentiment-count-chart"),
        dcc.Graph(id="sentiment-share-chart"),
        dcc.Graph(id="confidence-chart"),

        html.H3("Latest Tweets"),
        html.Div(id="latest-tweets"),

        dcc.Interval(
            id="live-update",
            interval=2000,
            n_intervals=0,
        ),
    ],
)


@app.callback(
    Output("summary-cards", "children"),
    Output("sentiment-count-chart", "figure"),
    Output("sentiment-share-chart", "figure"),
    Output("confidence-chart", "figure"),
    Output("latest-tweets", "children"),
    Input("live-update", "n_intervals"),
)
def update_dashboard(n):
    df = get_dataframe()

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Waiting for live tweets...",
            template="plotly_white",
        )

        cards = [
            make_card("Total tweets", "0"),
            make_card("Positive", "0"),
            make_card("Neutral", "0"),
            make_card("Negative", "0"),
        ]

        return cards, empty_fig, empty_fig, empty_fig, [html.Div("No tweets yet.")]

    total = len(df)
    positive = len(df[df["sentiment"] == "positive"])
    neutral = len(df[df["sentiment"] == "neutral"])
    negative = len(df[df["sentiment"] == "negative"])

    cards = [
        make_card("Total tweets", total),
        make_card("Positive", positive),
        make_card("Neutral", neutral),
        make_card("Negative", negative),
    ]

    # Sentiment count over time
    ts_count = make_sentiment_timeseries(df, "1min")

    count_fig = px.line(
        ts_count,
        x="timestamp",
        y="count",
        color="sentiment",
        title="Tweet sentiment volume over time",
        markers=True,
    )

    count_fig.update_layout(template="plotly_white")
    count_fig = add_goal_markers(count_fig)

    # Sentiment share over time
    ts_share = make_sentiment_share(df, "1min")

    share_fig = px.area(
        ts_share,
        x="timestamp",
        y="share",
        color="sentiment",
        title="Sentiment share over time",
    )

    share_fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
    )

    share_fig = add_goal_markers(share_fig)

    # Confidence distribution
    confidence_fig = px.box(
        df,
        x="sentiment",
        y="confidence",
        title="Sentiment model confidence by class",
    )

    confidence_fig.update_layout(template="plotly_white")

    latest_tweets = get_latest_tweets(df, 10)

    return cards, count_fig, share_fig, confidence_fig, latest_tweets


def make_card(title, value):
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "14px",
                    "color": "#555",
                },
            ),
            html.Div(
                str(value),
                style={
                    "fontSize": "28px",
                    "fontWeight": "bold",
                },
            ),
        ],
        style={
            "backgroundColor": "white",
            "padding": "16px",
            "borderRadius": "10px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",
            "width": "180px",
        },
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    stream_thread = threading.Thread(target=start_stream, daemon=True)
    stream_thread.start()

    app.run_server(debug=True, port=8050)
