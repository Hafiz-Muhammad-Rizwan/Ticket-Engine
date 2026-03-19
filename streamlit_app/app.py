from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Hybrid Ticket Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "customer_support_tickets.csv"
FALLBACK_DATA_PATHS = [
    APP_DIR / "artifact" / "corpus.csv",
    APP_DIR / "artifacts" / "corpus.csv",
    ROOT_DIR / "artifact" / "corpus.csv",
    ROOT_DIR / "artifacts" / "corpus.csv",
]
ARTIFACT_DIRS = [
    APP_DIR / "artifact",
    APP_DIR / "artifacts",
    ROOT_DIR / "streamlit_app" / "artifact",
    ROOT_DIR / "streamlit_app" / "artifacts",
    ROOT_DIR / "artifacts",
    ROOT_DIR / "artifact",
]

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def inject_modern_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&display=swap');

        :root {
            --ink-900: #0b1b33;
            --ink-700: #2b4564;
            --gold-500: #c79a3b;
            --mint-500: #0ea57a;
            --teal-500: #0f8d9a;
            --blue-500: #2a6fe3;
            --paper: rgba(255, 255, 255, 0.88);
        }

        .stApp {
            font-family: 'Sora', sans-serif;
            background:
                radial-gradient(1100px 520px at 9% -14%, rgba(14, 165, 122, 0.25), transparent 60%),
                radial-gradient(1000px 430px at 94% -8%, rgba(42, 111, 227, 0.21), transparent 56%),
                radial-gradient(800px 320px at 70% 110%, rgba(199, 154, 59, 0.15), transparent 58%),
                linear-gradient(160deg, #f6fffb 0%, #f1f8ff 48%, #ecf2ff 100%);
            color: var(--ink-900);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(247,253,255,0.95));
            border-right: 1px solid rgba(11, 27, 51, 0.10);
        }

        [data-testid="stAppViewContainer"] {
            animation: fadein 0.35s ease-in;
        }

        @keyframes fadein {
            from { opacity: 0.0; transform: translateY(6px); }
            to { opacity: 1.0; transform: translateY(0); }
        }

        .hero {
            border: 1px solid rgba(16, 42, 67, 0.10);
            border-radius: 22px;
            padding: 24px 26px;
            background:
                linear-gradient(105deg, rgba(255,255,255,0.95), rgba(255,255,255,0.78)),
                radial-gradient(400px 140px at 10% 0%, rgba(14,165,122,0.08), transparent 60%);
            backdrop-filter: blur(3px);
            box-shadow: 0 18px 34px rgba(12, 32, 59, 0.11);
            margin-bottom: 14px;
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: '';
            position: absolute;
            width: 180px;
            height: 180px;
            right: -40px;
            top: -55px;
            background: radial-gradient(circle, rgba(199,154,59,0.22), rgba(199,154,59,0.0) 70%);
            pointer-events: none;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            letter-spacing: -0.02em;
            color: var(--ink-900);
        }

        .hero p {
            margin: 8px 0 0 0;
            color: var(--ink-700);
            font-size: 1rem;
        }

        .metric-card {
            border: 1px solid rgba(10, 37, 64, 0.12);
            border-radius: 16px;
            padding: 14px;
            background: var(--paper);
            box-shadow: 0 16px 24px rgba(10, 37, 64, 0.09);
            min-height: 96px;
        }

        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #46627a;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--ink-900);
            line-height: 1.2;
        }

        .res-card {
            border-left: 5px solid var(--mint-500);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 12px 18px rgba(16, 42, 67, 0.08);
        }

        .soft-note {
            border: 1px dashed rgba(16, 42, 67, 0.28);
            border-radius: 12px;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.75);
            color: #334e68;
            font-size: 0.9rem;
        }

        .status-note {
            border: 1px solid rgba(16, 42, 67, 0.12);
            background: rgba(255, 255, 255, 0.88);
            border-radius: 12px;
            padding: 10px 12px;
            color: #334e68;
            margin-top: 8px;
        }

        [data-testid="stButton"] > button {
            border-radius: 12px;
            border: 1px solid rgba(16, 42, 67, 0.20);
            background: linear-gradient(135deg, #ffffff, #f2fbff);
            color: #17324d;
            font-weight: 600;
            padding-top: 0.56rem;
            padding-bottom: 0.56rem;
            transition: all 0.2s ease;
        }

        [data-testid="stButton"] > button:hover {
            border-color: rgba(14, 165, 122, 0.45);
            transform: translateY(-1px);
            box-shadow: 0 10px 16px rgba(16, 42, 67, 0.12);
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--teal-500);
            border: 2px solid #ffffff;
            box-shadow: 0 2px 8px rgba(15, 141, 154, 0.35);
        }

        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background: linear-gradient(90deg, var(--mint-500), var(--blue-500));
        }

        [data-testid="stSelectbox"] > div > div {
            border-radius: 10px;
            border: 1px solid rgba(11, 27, 51, 0.16);
            background: rgba(255, 255, 255, 0.90);
        }

        [data-testid="stTextArea"] textarea {
            border-radius: 12px;
            border: 1px solid rgba(11, 27, 51, 0.18);
            background: rgba(255, 255, 255, 0.92);
        }

        [data-testid="stTextArea"] textarea:focus {
            border-color: rgba(14, 165, 122, 0.55);
            box-shadow: 0 0 0 1px rgba(14, 165, 122, 0.30);
        }

        [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        [data-baseweb="tab"] {
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid rgba(16, 42, 67, 0.10);
        }

        [data-baseweb="tab-highlight"] {
            background-color: #138f71;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_note(message: str) -> None:
    st.markdown(f'<div class="status-note">{message}</div>', unsafe_allow_html=True)


def normalize_col_name(name: str) -> str:
    return str(name).strip().lower()


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    cols = list(df.columns)
    norm_map = {normalize_col_name(c): c for c in cols}

    def pick(candidates: List[str]) -> str | None:
        for c in candidates:
            if c in norm_map:
                return norm_map[c]
        return None

    text_col = pick(["ticket description", "description", "text", "message", "query", "issue"])
    type_col = pick(["ticket type", "type", "category", "label", "target", "class"])
    channel_col = pick(["ticket channel", "channel", "source", "contact channel"])
    resolution_col = pick(["resolution", "resolution details", "solution", "response", "resolution summary"])

    if text_col is None:
        obj_cols = [c for c in cols if df[c].dtype == "object"]
        if not obj_cols:
            raise ValueError("No text-like column found in dataset.")
        text_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())

    if type_col is None:
        obj_cols = [c for c in cols if df[c].dtype == "object" and c != text_col]
        if not obj_cols:
            raise ValueError("No ticket type column found in dataset.")
        type_col = min(obj_cols, key=lambda c: abs(df[c].nunique(dropna=True) - 6))

    if channel_col is None:
        obj_cols = [c for c in cols if df[c].dtype == "object" and c not in [text_col, type_col]]
        channel_col = obj_cols[0] if obj_cols else type_col

    if resolution_col is None:
        resolution_col = text_col

    return text_col, type_col, channel_col, resolution_col


def regex_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def full_tokens(text: str) -> List[str]:
    uni = regex_tokenize(text)
    return uni + generate_ngrams(uni, 2) + generate_ngrams(uni, 3)


def safe_resolution_text(value: str) -> str:
    v = str(value).strip()
    if v.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return "No resolution available"
    return v


@dataclass
class SearchArtifacts:
    data: pd.DataFrame
    text_col: str
    type_col: str
    channel_col: str
    resolution_col: str
    vocab: List[str]
    stoi: Dict[str, int]
    idf: np.ndarray
    embed_matrix: np.ndarray
    postings: Dict[int, List[Tuple[int, float]]]
    doc_norm: np.ndarray
    dense_doc_vectors: np.ndarray
    fallback_semantic: bool
    source_dir: str


class HybridEngine:
    def __init__(self, artifacts: SearchArtifacts) -> None:
        self.a = artifacts
        self.unk_idx = self.a.stoi.get("<UNK>", 1)

    def query_tfidf(self, query: str) -> Tuple[Dict[int, float], float]:
        q_tokens = full_tokens(query)
        q_idxs = [self.a.stoi.get(t, self.unk_idx) for t in q_tokens]
        q_tf = Counter(q_idxs)
        q_weights = {idx: float(freq) * float(self.a.idf[idx]) for idx, freq in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_weights.values()) + 1e-12)
        return q_weights, q_norm

    def tfidf_similarity(self, query: str) -> np.ndarray:
        q_weights, q_norm = self.query_tfidf(query)
        sims = np.zeros(len(self.a.data), dtype=np.float32)
        for term_idx, qv in q_weights.items():
            for doc_idx, dv in self.a.postings.get(int(term_idx), []):
                sims[doc_idx] += float(qv) * float(dv)
        return sims / (self.a.doc_norm * q_norm)

    def dense_vector(self, text: str) -> np.ndarray:
        tokens = regex_tokenize(text)
        if not tokens:
            return np.zeros((self.a.embed_matrix.shape[1],), dtype=np.float32)

        tf = Counter(tokens)
        weighted_sum = np.zeros((self.a.embed_matrix.shape[1],), dtype=np.float32)
        total_weight = 0.0

        for tok, freq in tf.items():
            idx = self.a.stoi.get(tok, self.unk_idx)
            w = float(freq) * float(self.a.idf[idx])
            weighted_sum += w * self.a.embed_matrix[idx]
            total_weight += w

        if total_weight <= 0:
            return np.zeros((self.a.embed_matrix.shape[1],), dtype=np.float32)
        return weighted_sum / total_weight

    def dense_similarity(self, query: str) -> np.ndarray:
        qv = self.dense_vector(query)
        qn = np.linalg.norm(qv) + 1e-12
        d = self.a.dense_doc_vectors
        dn = np.linalg.norm(d, axis=1) + 1e-12
        return (d @ qv) / (dn * qn)

    def retrieve(self, query: str, alpha: float = 0.6, top_k: int = 5) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        tfidf_s = self.tfidf_similarity(query)
        dense_s = self.dense_similarity(query)
        hybrid = alpha * tfidf_s + (1.0 - alpha) * dense_s
        idxs = np.argsort(-hybrid)[:top_k]

        rows = []
        for rank, i in enumerate(idxs, 1):
            r = self.a.data.iloc[int(i)]
            rows.append(
                {
                    "rank": rank,
                    "ticket_type": str(r[self.a.type_col]),
                    "channel": str(r[self.a.channel_col]),
                    "description": str(r[self.a.text_col]),
                    "resolution": safe_resolution_text(str(r[self.a.resolution_col])),
                    "tfidf_score": float(tfidf_s[i]),
                    "semantic_score": float(dense_s[i]),
                    "hybrid_score": float(hybrid[i]),
                }
            )

        return pd.DataFrame(rows), tfidf_s, dense_s, hybrid


def load_from_artifacts() -> SearchArtifacts | None:
    for d in ARTIFACT_DIRS:
        meta_path = d / "metadata.json"
        corpus_path = d / "corpus.csv"
        idf_path = d / "idf.npy"
        embed_path = d / "embed_matrix.npy"
        stoi_path = d / "stoi.json"
        dense_path = d / "dense_doc_vectors.npy"

        required = [meta_path, corpus_path, idf_path, embed_path, stoi_path, dense_path]
        if not all(p.exists() for p in required):
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = pd.read_csv(corpus_path)
        text_col = meta["text_col"]
        type_col = meta["type_col"]
        channel_col = meta["channel_col"]
        resolution_col = meta["resolution_col"]

        stoi = json.loads(stoi_path.read_text(encoding="utf-8"))
        vocab = [None] * len(stoi)
        for tok, idx in stoi.items():
            vocab[int(idx)] = tok

        idf = np.load(idf_path)
        embed_matrix = np.load(embed_path)
        dense_doc_vectors = np.load(dense_path)

        data = data.copy()
        data[resolution_col] = data[resolution_col].astype(str).map(safe_resolution_text)

        docs_all = [full_tokens(t) for t in data[text_col].astype(str).tolist()]
        unk_idx = stoi.get("<UNK>", 1)

        rows, cols, vals = [], [], []
        for r, toks in enumerate(docs_all):
            tf = Counter(stoi.get(t, unk_idx) for t in toks)
            for c, f in tf.items():
                rows.append(r)
                cols.append(c)
                vals.append(float(f) * float(idf[c]))

        postings = defaultdict(list)
        doc_norm_sq = np.zeros(len(data), dtype=np.float32)
        for r, c, v in zip(rows, cols, vals):
            postings[int(c)].append((int(r), float(v)))
            doc_norm_sq[int(r)] += float(v) * float(v)

        return SearchArtifacts(
            data=data,
            text_col=text_col,
            type_col=type_col,
            channel_col=channel_col,
            resolution_col=resolution_col,
            vocab=vocab,
            stoi=stoi,
            idf=idf,
            embed_matrix=embed_matrix,
            postings=dict(postings),
            doc_norm=np.sqrt(np.maximum(doc_norm_sq, 1e-12)),
            dense_doc_vectors=dense_doc_vectors,
            fallback_semantic=False,
            source_dir=str(d),
        )

    return None


def build_from_csv(csv_path: Path) -> SearchArtifacts:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    text_col, type_col, channel_col, resolution_col = detect_columns(df)

    data = df[[text_col, type_col, channel_col, resolution_col]].copy()
    data = data.dropna(subset=[text_col, type_col, channel_col]).reset_index(drop=True)

    data[text_col] = data[text_col].astype(str).str.strip()
    data[text_col] = data[text_col].str.replace(r"\{[^}]*\}", " ", regex=True)
    data[text_col] = data[text_col].str.replace(r"\s+", " ", regex=True).str.strip()
    data[type_col] = data[type_col].astype(str).str.strip()
    data[channel_col] = data[channel_col].astype(str).str.strip()
    data[resolution_col] = data[resolution_col].astype(str).map(safe_resolution_text)
    data = data[data[text_col] != ""].reset_index(drop=True)

    docs_uni = [regex_tokenize(t) for t in data[text_col].tolist()]
    docs_all = [full_tokens(t) for t in data[text_col].tolist()]

    max_vocab = 5000
    specials = ["<PAD>", "<UNK>"]
    counter = Counter()
    for toks in docs_all:
        counter.update(toks)

    most_common = [w for w, _ in counter.most_common(max_vocab - len(specials))]
    vocab = specials + most_common
    stoi = {w: i for i, w in enumerate(vocab)}
    unk_idx = stoi["<UNK>"]

    n_docs = len(docs_all)
    v_size = len(vocab)

    docs_idx_all = [[stoi.get(t, unk_idx) for t in toks] for toks in docs_all]
    docs_idx_uni = [[stoi.get(t, unk_idx) for t in toks] for toks in docs_uni]

    df_count = np.zeros(v_size, dtype=np.int64)
    for idxs in docs_idx_all:
        for u in set(idxs):
            df_count[u] += 1

    idf = np.log((n_docs + 1) / (df_count + 1)) + 1.0

    rows, cols, vals = [], [], []
    for r, idxs in enumerate(docs_idx_all):
        tf = Counter(idxs)
        for c, freq in tf.items():
            w = float(freq) * float(idf[c])
            rows.append(r)
            cols.append(c)
            vals.append(w)

    postings = defaultdict(list)
    doc_norm_sq = np.zeros(n_docs, dtype=np.float32)
    for r, c, v in zip(rows, cols, vals):
        postings[int(c)].append((int(r), float(v)))
        doc_norm_sq[int(r)] += float(v) * float(v)

    # Fallback semantic vectors if notebook artifacts are unavailable.
    emb_dim = 300
    embed_matrix = np.zeros((v_size, emb_dim), dtype=np.float32)
    rng = np.random.default_rng(42)
    embed_matrix[unk_idx] = rng.normal(0, 0.1, size=(emb_dim,)).astype(np.float32)
    for tok, idx in stoi.items():
        if idx == 0:
            continue
        seed = abs(hash(tok)) % (2**32)
        token_rng = np.random.default_rng(seed)
        embed_matrix[idx] = token_rng.normal(0, 0.08, size=(emb_dim,)).astype(np.float32)

    dense_doc_vectors = np.zeros((n_docs, emb_dim), dtype=np.float32)
    for i, idxs in enumerate(docs_idx_uni):
        if not idxs:
            continue
        tf = Counter(idxs)
        w_sum = np.zeros((emb_dim,), dtype=np.float32)
        w_total = 0.0
        for idx, freq in tf.items():
            w = float(freq) * float(idf[idx])
            w_sum += w * embed_matrix[idx]
            w_total += w
        if w_total > 0:
            dense_doc_vectors[i] = w_sum / w_total

    return SearchArtifacts(
        data=data,
        text_col=text_col,
        type_col=type_col,
        channel_col=channel_col,
        resolution_col=resolution_col,
        vocab=vocab,
        stoi=stoi,
        idf=idf,
        embed_matrix=embed_matrix,
        postings=dict(postings),
        doc_norm=np.sqrt(np.maximum(doc_norm_sq, 1e-12)),
        dense_doc_vectors=dense_doc_vectors,
        fallback_semantic=True,
        source_dir=str(csv_path),
    )


@st.cache_resource(show_spinner=True)
def load_engine() -> HybridEngine:
    artifacts = load_from_artifacts()
    if artifacts is None:
        csv_candidates = [DEFAULT_DATA_PATH] + FALLBACK_DATA_PATHS
        csv_path = next((p for p in csv_candidates if p.exists()), DEFAULT_DATA_PATH)
        artifacts = build_from_csv(csv_path)
    return HybridEngine(artifacts)


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_modern_css()
    engine = load_engine()
    a = engine.a

    st.markdown(
        """
        <div class="hero">
            <h1>Hybrid Ticket Intelligence Studio</h1>
            <p>Blend keyword precision with semantic context, tune alpha in real time, and inspect actionable historical resolutions in a premium decision workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Search Controls")
        alpha = st.slider("Alpha (Keyword vs Semantic)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
        top_k = st.slider("Top Results", min_value=3, max_value=10, value=5, step=1)
        st.caption("Alpha=1.0 means pure TF-IDF. Alpha=0.0 means pure semantic.")

        st.divider()
        st.subheader("System")
        st.write(f"Corpus size: **{len(a.data):,}**")
        st.write(f"Vocabulary size: **{len(a.vocab):,}**")
        st.write(f"Embedding dim: **{a.embed_matrix.shape[1]}**")
        source_note = "Notebook artifacts" if not a.fallback_semantic else "Fallback semantic vectors"
        st.write(f"Semantic source: **{source_note}**")
        st.caption(f"Loaded from: {a.source_dir}")
        st.caption("Clean interface mode: no decorative icons")

    default_query = "My billing payment failed and account is blocked."
    query = st.text_area(
        "Describe a new customer ticket",
        value=default_query,
        height=110,
        placeholder="Type a ticket issue here...",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        run_btn = st.button("Run Hybrid Search", use_container_width=True)
    with col_b:
        sample_btn = st.button("Use Random Real Ticket", use_container_width=True)

    if sample_btn:
        query = str(a.data.sample(1, random_state=42)[a.text_col].iloc[0])

    if run_btn or sample_btn:
        if not query.strip():
            render_status_note("Please enter a ticket description before running the search.")
            return

        results, tfidf_s, dense_s, hybrid_s = engine.retrieve(query, alpha=alpha, top_k=top_k)
        top1 = results.iloc[0]

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric("Predicted Ticket Type", str(top1["ticket_type"]))
        with m2:
            render_metric("Top Hybrid Score", f"{top1['hybrid_score']:.4f}")
        with m3:
            render_metric("Top Channel", str(top1["channel"]))

        st.markdown("### Top 3 Similar Resolutions")
        for i in range(min(3, len(results))):
            r = results.iloc[i]
            st.markdown(
                f"""
                <div class="res-card">
                    <strong>#{i+1} - {r['ticket_type']} ({r['channel']})</strong><br/>
                    {r['resolution']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        tab1, tab2, tab3 = st.tabs(["Hybrid Results", "Model Blend", "Diagnostics"])

        with tab1:
            show_df = results[[
                "rank",
                "ticket_type",
                "channel",
                "tfidf_score",
                "semantic_score",
                "hybrid_score",
                "description",
            ]].copy()
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        with tab2:
            cmp = pd.DataFrame(
                {
                    "Method": ["TF-IDF", "Semantic", "Hybrid"],
                    "Score": [float(top1["tfidf_score"]), float(top1["semantic_score"]), float(top1["hybrid_score"])],
                }
            )
            fig = px.bar(
                cmp,
                x="Method",
                y="Score",
                color="Method",
                title="Top-Match Score Composition",
                color_discrete_sequence=["#1f9d55", "#2f80ed", "#0b4f6c"],
            )
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            blend_line = pd.DataFrame(
                {
                    "alpha": np.round(np.linspace(0.0, 1.0, 21), 2),
                }
            )
            blend_line["expected_score"] = (
                blend_line["alpha"] * float(top1["tfidf_score"]) + (1.0 - blend_line["alpha"]) * float(top1["semantic_score"])
            )
            fig_line = px.line(
                blend_line,
                x="alpha",
                y="expected_score",
                title="Score Trajectory Across Alpha",
            )
            fig_line.update_traces(line=dict(width=3, color="#148f77"))
            fig_line.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown(
                f"""
                <div class="soft-note">
                <strong>Blend Explanation:</strong> Hybrid score = {alpha:.2f} * TF-IDF + {1-alpha:.2f} * Semantic.
                Move alpha in the sidebar to watch ranking behavior change in real time.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab3:
            type_counts = a.data[a.type_col].value_counts().head(8).reset_index()
            type_counts.columns = ["Ticket Type", "Count"]
            fig2 = px.bar(
                type_counts,
                x="Count",
                y="Ticket Type",
                orientation="h",
                title="Top Ticket Types in Corpus",
                color="Count",
                color_continuous_scale="Tealgrn",
            )
            fig2.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig2, use_container_width=True)

            st.write("Data columns used by the app:")
            st.code(
                f"text_col={a.text_col}\n"
                f"type_col={a.type_col}\n"
                f"channel_col={a.channel_col}\n"
                f"resolution_col={a.resolution_col}"
            )

            if a.fallback_semantic:
                render_status_note(
                    "Artifacts from Kaggle were not found locally. "
                    "The app is running with fallback semantic vectors. "
                    "For best quality, place notebook artifacts under streamlit_app/artifacts."
                )
            else:
                render_status_note("Notebook artifacts detected. Running with your exported embeddings and mappings.")

    else:
        st.markdown(
            """
            <div class="soft-note">
            Enter a ticket description, adjust the alpha slider, and click <strong>Run Hybrid Search</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
