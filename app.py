"""
CineMatch - Movie Recommendation System (Streamlit)
Uses: movie_list.pkl, similarity.pkl  (+ optional tmdb_5000_movies.csv for details)
Run :  streamlit run app.py
"""
import hashlib
import html
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MOVIE_PKL = BASE_DIR / "movie_list.pkl"
SIM_PKL = BASE_DIR / "similarity.pkl"
MOVIES_CSV = BASE_DIR / "tmdb_5000_movies.csv"  # optional (ratings, genres, overview)

# ----------------------------------------------------------------------------
# Styling + animations
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }

/* animated dark background */
.stApp {
    background: radial-gradient(circle at 15% 10%, #1b1038 0%, transparent 45%),
                radial-gradient(circle at 85% 20%, #3a0f2e 0%, transparent 45%),
                radial-gradient(circle at 50% 100%, #0b1d3a 0%, transparent 50%),
                #07070d;
    background-attachment: fixed;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1250px;}

/* floating blobs */
.blob {position: fixed; border-radius: 50%; filter: blur(80px); opacity: .28;
       z-index: 0; animation: float 14s ease-in-out infinite; pointer-events: none;}
.b1 {width: 340px; height: 340px; background: #e50914; top: -80px; left: -80px;}
.b2 {width: 300px; height: 300px; background: #6c2bd9; bottom: -60px; right: -60px;
     animation-delay: -6s;}
@keyframes float {
    0%,100% {transform: translate(0,0) scale(1);}
    50%     {transform: translate(40px,-30px) scale(1.15);}
}

/* hero */
.hero {text-align: center; padding: 1.2rem 0 .4rem; animation: fadeDown .9s ease both;}
.hero h1 {
    font-size: clamp(2.4rem, 6vw, 4.2rem); font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #ff4d4d, #ff9a3c, #ff4d9d, #8a5cff, #ff4d4d);
    background-size: 300% 100%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shine 7s linear infinite;
}
.hero p {color: #a9a9c0; font-weight: 300; font-size: 1.1rem; margin-top: .3rem;}
@keyframes shine {to {background-position: 300% 0;}}
@keyframes fadeDown {from {opacity: 0; transform: translateY(-25px);}
                     to {opacity: 1; transform: translateY(0);}}

/* stat chips */
.chips {display: flex; justify-content: center; gap: .8rem; flex-wrap: wrap; margin: 1rem 0 1.4rem;}
.chip {background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
       padding: .35rem .9rem; border-radius: 999px; color: #cfcfe6; font-size: .85rem;
       backdrop-filter: blur(8px);}

/* inputs */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.06) !important; border: 1px solid rgba(255,255,255,.15) !important;
    border-radius: 14px !important; color: #fff !important;
}
.stButton > button {
    width: 100%; border: none; color: #fff; font-weight: 600; letter-spacing: .5px;
    padding: .7rem 1rem; border-radius: 14px;
    background: linear-gradient(90deg, #e50914, #ff5e62);
    box-shadow: 0 8px 24px rgba(229,9,20,.35);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover {transform: translateY(-3px) scale(1.02);
                          box-shadow: 0 14px 30px rgba(229,9,20,.55); color: #fff;}
.stButton > button:active {transform: scale(.98);}

/* section title */
.sec-title {font-size: 1.5rem; font-weight: 600; color: #fff; margin: 1.6rem 0 1rem;
            animation: fadeUp .6s ease both;}
.sec-title span {color: #ff5e62;}

/* movie grid + cards (no posters: colourful, information-rich cards) */
.grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1.4rem;}
.card {
    position: relative; border-radius: 20px; overflow: hidden; padding: 1.2rem 1.2rem 1.1rem;
    min-height: 300px; display: flex; flex-direction: column;
    background: linear-gradient(145deg, var(--c1), var(--c2));
    border: 1px solid rgba(255,255,255,.14);
    opacity: 0; animation: fadeUp .7s ease forwards;
    transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease;
}
.card::before {content: ""; position: absolute; inset: 0; background: rgba(8,8,16,.55);}
.card::after {content: ""; position: absolute; top: -60%; left: -60%; width: 60%; height: 220%;
              background: linear-gradient(90deg, transparent, rgba(255,255,255,.14), transparent);
              transform: rotate(20deg); transition: left .8s ease;}
.card:hover {transform: translateY(-10px) scale(1.03);
             box-shadow: 0 22px 45px rgba(229,9,20,.3); z-index: 5;}
.card:hover::after {left: 130%;}
.card > * {position: relative; z-index: 2;}
.top {display: flex; justify-content: space-between; align-items: center;}
.rank {background: rgba(0,0,0,.55); color: #fff; font-weight: 800; border-radius: 10px;
       padding: 2px 11px; backdrop-filter: blur(6px);}
.emoji {font-size: 3.2rem; text-align: center; margin: .9rem 0 .5rem;
        filter: drop-shadow(0 6px 12px rgba(0,0,0,.5)); transition: transform .4s ease;}
.card:hover .emoji {transform: scale(1.18) rotate(-6deg);}
.card h4 {margin: 0; color: #fff; font-size: 1.1rem; font-weight: 600; line-height: 1.25;
          text-align: center;}
.meta {color: #ffd76a; font-size: .82rem; margin-top: 5px; text-align: center;}
.genres {display: flex; gap: .35rem; justify-content: center; flex-wrap: wrap; margin-top: .55rem;}
.genres span {font-size: .68rem; color: #fff; background: rgba(255,255,255,.16);
              padding: 2px 9px; border-radius: 999px;}
.over {color: #d8d8ea; font-size: .75rem; line-height: 1.45; margin-top: .7rem; text-align: center;}
.bar-wrap {margin-top: auto; padding-top: .9rem;}
.bar-label {display: flex; justify-content: space-between; color: #fff; font-size: .72rem;
            font-weight: 600; margin-bottom: 4px;}
.bar {height: 7px; border-radius: 99px; background: rgba(255,255,255,.18); overflow: hidden;}
.bar > div {height: 100%; width: 0; border-radius: 99px;
            background: linear-gradient(90deg, #ffd200, #ff5e62);
            animation: fill 1.2s .4s cubic-bezier(.2,.8,.2,1) forwards;}
@keyframes fill {to {width: var(--w);}}
@keyframes fadeUp {from {opacity: 0; transform: translateY(30px);}
                   to {opacity: 1; transform: translateY(0);}}

/* selected movie panel */
.picked {display: flex; gap: 1.4rem; align-items: center; padding: 1rem; border-radius: 18px;
         background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
         animation: fadeUp .6s ease both; backdrop-filter: blur(10px);}
.picked .big {font-size: 3.4rem; width: 90px; text-align: center;}
.picked h3 {margin: 0 0 .2rem; color: #fff;}
.picked p {margin: .2rem 0 0; color: #b9b9d0; font-size: .88rem;}

/* sidebar */
section[data-testid="stSidebar"] {background: rgba(10,10,18,.92);
    border-right: 1px solid rgba(255,255,255,.08);}
label, .stSlider label {color: #cfcfe6 !important;}
.foot {text-align: center; color: #6f6f88; font-size: .8rem; margin: 3rem 0 1rem;}
</style>
<div class="blob b1"></div><div class="blob b2"></div>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    with open(MOVIE_PKL, "rb") as f:
        movies = pickle.load(f).reset_index(drop=True)
    with open(SIM_PKL, "rb") as f:
        sim = pickle.load(f)
    return movies, np.asarray(sim, dtype=np.float32)  # float32 halves memory use


@st.cache_data(show_spinner=False)
def load_metadata():
    """Optional extra info (rating, year, genres, overview) keyed by TMDB id."""
    if not MOVIES_CSV.exists():
        return {}
    df = pd.read_csv(
        MOVIES_CSV,
        usecols=["id", "genres", "overview", "release_date", "vote_average"],
    )
    meta = {}
    for r in df.itertuples(index=False):
        try:
            genres = [g["name"] for g in json.loads(r.genres)][:3]
        except Exception:
            genres = []
        year = str(r.release_date)[:4] if isinstance(r.release_date, str) else ""
        meta[int(r.id)] = {
            "genres": genres,
            "overview": r.overview if isinstance(r.overview, str) else "",
            "year": year,
            "rating": float(r.vote_average) if pd.notna(r.vote_average) else None,
        }
    return meta


# ----------------------------------------------------------------------------
# Recommendation logic (same idea as your notebook's recommend())
# ----------------------------------------------------------------------------
def recommend(title: str, n: int, movies: pd.DataFrame, sim: np.ndarray):
    idx = int(np.where(movies["title"].values == title)[0][0])
    scores = sim[idx]
    order = np.argsort(-scores)
    order = [i for i in order if i != idx][:n]
    return [
        {
            "id": int(movies.iloc[i]["id"]),
            "title": movies.iloc[i]["title"],
            "score": float(scores[i]),
        }
        for i in order
    ]


GENRE_EMOJI = {
    "Action": "💥", "Adventure": "🧭", "Animation": "🎨", "Comedy": "😂", "Crime": "🕵️",
    "Documentary": "🎞️", "Drama": "🎭", "Family": "👨‍👩‍👧", "Fantasy": "🧙", "History": "📜",
    "Horror": "👻", "Music": "🎵", "Mystery": "🔍", "Romance": "❤️", "Science Fiction": "🚀",
    "Thriller": "😱", "War": "⚔️", "Western": "🤠",
}


def emoji_for(meta: dict) -> str:
    for g in meta.get("genres", []):
        if g in GENRE_EMOJI:
            return GENRE_EMOJI[g]
    return "🎬"


def colors_for(title: str) -> tuple[str, str]:
    """Stable, unique-looking gradient per movie (derived from its title)."""
    h = int(hashlib.md5(title.encode()).hexdigest()[:6], 16) % 360
    return f"hsl({h},65%,38%)", f"hsl({(h + 55) % 360},70%,28%)"


def card_html(rank: int, item: dict, meta: dict) -> str:
    title = html.escape(item["title"])
    m = meta.get(item["id"], {})
    c1, c2 = colors_for(item["title"])

    bits = []
    if m.get("rating"):
        bits.append(f"⭐ {m['rating']:.1f}")
    if m.get("year"):
        bits.append(m["year"])
    meta_line = f'<div class="meta">{" · ".join(bits)}</div>' if bits else ""
    genres = "".join(f"<span>{html.escape(g)}</span>" for g in m.get("genres", []))
    genres = f'<div class="genres">{genres}</div>' if genres else ""
    ov = m.get("overview", "")
    over = f'<div class="over">{html.escape(ov[:130])}{"…" if len(ov) > 130 else ""}</div>' if ov else ""

    pct = max(0, min(100, round(item["score"] * 100)))
    return f"""
    <div class="card" style="--c1:{c1};--c2:{c2};animation-delay:{rank * 0.12:.2f}s">
        <div class="top"><div class="rank">#{rank}</div></div>
        <div class="emoji">{emoji_for(m)}</div>
        <h4>{title}</h4>{meta_line}{genres}{over}
        <div class="bar-wrap">
            <div class="bar-label"><span>Similarity</span><span>{pct}% match</span></div>
            <div class="bar"><div style="--w:{pct}%"></div></div>
        </div>
    </div>"""


# ----------------------------------------------------------------------------
# Guard: required files
# ----------------------------------------------------------------------------
missing = [p.name for p in (MOVIE_PKL, SIM_PKL) if not p.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Place them in the same folder as app.py.")
    st.stop()

with st.spinner("Loading the cinema reels..."):
    movies_df, similarity = load_model()
metadata = load_metadata()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    n_recs = st.slider("Number of recommendations", 3, 15, 5)
    st.markdown("---")
    st.markdown(
        "**How it works**  \n"
        "Each movie's overview, genres, keywords and director are turned into a "
        "tag vector (CountVectorizer + Porter stemming). Cosine similarity then "
        "finds the closest movies."
    )

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🎬 CineMatch</h1>
        <p>Tell us a movie you love, we'll find your next favourite.</p>
    </div>
    <div class="chips">
        <div class="chip">🎞️ {len(movies_df):,} movies</div>
        <div class="chip">🧠 Content-based filtering</div>
        <div class="chip">📐 Cosine similarity</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Search area
# ----------------------------------------------------------------------------
titles = sorted(movies_df["title"].unique().tolist())
default_idx = titles.index("Avatar") if "Avatar" in titles else 0

_, mid, _ = st.columns([1, 3, 1])
with mid:
    selected = st.selectbox(
        "🔍 Search or pick a movie",
        titles,
        index=default_idx,
        placeholder="Type a movie name...",
    )
    go = st.button("✨ Recommend Movies")

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
if go:
    with st.spinner("Rolling the film... finding similar movies 🍿"):
        recs = recommend(selected, n_recs, movies_df, similarity)
        sel_id = int(movies_df.loc[movies_df["title"] == selected, "id"].iloc[0])

    # selected movie panel
    sm = metadata.get(sel_id, {})
    sbits = " · ".join(
        b for b in [f"⭐ {sm['rating']:.1f}" if sm.get("rating") else "", sm.get("year", ""),
                    ", ".join(sm.get("genres", []))] if b
    )
    st.markdown(
        f"""<div class="picked"><div class="big">{emoji_for(sm)}</div><div>
            <div style="color:#ff5e62;font-size:.8rem;font-weight:600;">YOU SELECTED</div>
            <h3>{html.escape(selected)}</h3>
            <p>{html.escape(sbits)}</p>
            <p>{html.escape(sm.get('overview', '')[:260])}</p>
        </div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="sec-title">Because you liked <span>{html.escape(selected)}</span>, you may enjoy</div>',
        unsafe_allow_html=True,
    )
    cards = "".join(
        card_html(i + 1, r, metadata) for i, r in enumerate(recs)
    )
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)
    st.toast(f"Found {len(recs)} movies for you!", icon="🍿")
else:
    st.markdown(
        '<p style="text-align:center;color:#6f6f88;margin-top:2.5rem;">'
        "Pick a movie above and hit <b>Recommend Movies</b> to begin 🎥</p>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">Built with Streamlit · Data from TMDB 5000 Movies</div>',
            unsafe_allow_html=True)