"""Generate the article's figures into figures/. Run eval_retrieval.py first.

    python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK, MUTED, ACCENT, ACCENT2, FILL, RED = "#1A1A1A", "#6B7280", "#2563EB", "#F59E0B", "#EEF2FF", "#DC2626"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def box(ax, x, y, w, h, text, fc=FILL, ec=ACCENT, size=15, weight="bold", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=color)


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2, color=color))


def cover(w_px, h_px, name):
    dpi = 100
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#0F172A")
    # motif: two score bars (retrieval / generation)
    for i, (label, val, c) in enumerate([("retrieval", 0.94, ACCENT), ("generation", 0.0, ACCENT2)]):
        y = 2.0 - i * 0.9
        ax.add_patch(FancyBboxPatch((1.2, y), 6.0, 0.42, boxstyle="round,pad=0,rounding_size=0.2", fc="#1E293B", ec="none"))
        ax.add_patch(FancyBboxPatch((1.2, y), 6.0 * (val if val else 0.78), 0.42,
                                    boxstyle="round,pad=0,rounding_size=0.2", fc=c, ec="none"))
        ax.text(7.5, y + 0.21, label, va="center", fontsize=20, color="#CBD5E1")
    ax.text(1.2, 6.4, "Stop Eyeballing", fontsize=64, weight="bold", color="white", va="center")
    ax.text(1.2, 5.1, "Your RAG", fontsize=64, weight="bold", color="white", va="center")
    ax.text(1.2, 3.75, "Measure retrieval and generation separately",
            fontsize=26, color="#94A3B8", va="center")
    fig.savefig(OUT / name, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=100)
    ax.set_xlim(0, 14)
    ax.set_ylim(1.3, 6.3)
    ax.axis("off")
    steps = [("Documents", "stale or\nmissing content"), ("Chunk &\nindex", "chunks split\nthe answer"),
             ("Retrieve\ntop-k", "right chunk\nnot found"), ("Generate\nanswer", "hallucinates or\nignores context")]
    xs = [0.4, 3.8, 7.2, 10.6]
    for (label, fail), x in zip(steps, xs):
        box(ax, x, 3.2, 3.0, 1.4, label)
        ax.text(x + 1.5, 2.55, "✗ " + fail.replace("\n", "\n   "), ha="center", va="top", fontsize=13, color=RED)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + 3.0, 3.9, b, 3.9)
    # metric brackets
    ax.plot([0.4, 10.2], [5.05, 5.05], color=ACCENT, lw=3)
    ax.text(5.3, 5.35, "Retrieval metrics: hit rate@k, recall@k, MRR", ha="center", fontsize=15, color=ACCENT, weight="bold")
    ax.plot([10.6, 13.6], [5.05, 5.05], color=ACCENT2, lw=3)
    ax.text(12.1, 5.85, "Generation metrics", ha="center", fontsize=15, color="#B45309", weight="bold")
    ax.text(12.1, 5.35, "faithfulness · relevance\n· correctness", ha="center", va="center", fontsize=12, color="#B45309")
    fig.savefig(OUT / "01-pipeline.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def retrieval_chart():
    res = json.loads((ROOT / "results" / "retrieval.json").read_text())
    names = list(res)
    short = [n.split(" (")[0] for n in names]
    kw = [res[n]["by_type"]["keyword"]["hit@3"] for n in names]
    pa = [res[n]["by_type"]["paraphrase"]["hit@3"] for n in names]
    n_kw = res[names[0]]["by_type"]["keyword"]["n"]
    n_pa = res[names[0]]["by_type"]["paraphrase"]["n"]
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=120)
    x = range(len(names))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], kw, w, color="#CBD5E1", label=f"Keyword-style questions (n={n_kw})")
    b2 = ax.bar([i + w / 2 for i in x], pa, w, color=ACCENT, label=f"Paraphrased questions (n={n_pa})")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.0%}",
                    ha="center", fontsize=15, weight="bold")
    ax.set_xticks(list(x), short, fontsize=16)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Hit rate@3 (right doc in top 3)")
    ax.set_title("Paraphrased questions expose the gap that keyword questions hide",
                 fontsize=18, weight="bold", loc="left", pad=16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=14)
    fig.savefig(OUT / "02-retrieval-results.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diagnosis():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=110)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    cells = [
        (2.2, 4.0, "#DCFCE7", "#16A34A", "Healthy", "Ship it, keep\nmonitoring"),
        (7.0, 4.0, "#FEF3C7", ACCENT2, "Generation problem", "Fix the prompt, model\nor context format"),
        (2.2, 0.4, "#FEF3C7", ACCENT2, "Lucky answers", "Model knows it anyway.\nFix retrieval before it bites"),
        (7.0, 0.4, "#FEE2E2", RED, "Retrieval problem", "Fix chunking, embeddings,\nquery rewriting, top-k"),
    ]
    for x, y, fc, ec, title, body in cells:
        ax.add_patch(FancyBboxPatch((x, y), 4.6, 3.4, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=2))
        ax.text(x + 2.3, y + 2.4, title, ha="center", fontsize=19, weight="bold")
        ax.text(x + 2.3, y + 1.2, body, ha="center", va="center", fontsize=15, color=INK)
    ax.text(4.5, 7.6, "Answer quality: high", ha="center", fontsize=15, color=MUTED, weight="bold")
    ax.text(9.3, 7.6, "Answer quality: low", ha="center", fontsize=15, color=MUTED, weight="bold")
    ax.text(1.7, 5.7, "Retrieval:\nhigh", ha="center", va="center", fontsize=15, color=MUTED, weight="bold", rotation=90)
    ax.text(1.7, 2.1, "Retrieval:\nlow", ha="center", va="center", fontsize=15, color=MUTED, weight="bold", rotation=90)
    fig.savefig(OUT / "04-diagnosis.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generation_chart():
    files = {"BM25": ROOT / "results" / "generation_bm25.json", "Embeddings": ROOT / "results" / "generation_embeddings.json"}
    if not all(f.exists() for f in files.values()):
        print("Skipping 03-generation-results.png (run eval_generation.py for bm25 and embeddings first)")
        return
    data = {k: json.loads(f.read_text()) for k, f in files.items()}
    metrics = [("Retrieval hit@3", None), ("Faithfulness", "faithfulness"), ("Correct answers", "correct")]

    def value(d, key):
        if key:
            return d["summary"][key]
        rows = [r for r in d["rows"] if r["relevant_ids"]]
        return sum(r["retrieval_hit"] for r in rows) / len(rows)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=120)
    w = 0.36
    for j, (name, color) in enumerate([("BM25", "#CBD5E1"), ("Embeddings", ACCENT)]):
        vals = [value(data[name], key) for _, key in metrics]
        bars = ax.bar([i + (j - 0.5) * w for i in range(len(metrics))], vals, w, color=color, label=f"{name} retriever")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.0%}",
                    ha="center", fontsize=15, weight="bold")
    ax.set_xticks(range(len(metrics)), [m for m, _ in metrics], fontsize=16)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Faithfulness stayed high while correctness dropped with weak retrieval",
                 fontsize=18, weight="bold", loc="left", pad=16)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=14)
    fig.savefig(OUT / "03-generation-results.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    if not (ROOT / "results" / "retrieval.json").exists():
        sys.exit("Run scripts/eval_retrieval.py first.")
    cover(1400, 788, "00-cover.png")
    cover(1920, 1080, "linkedin-cover.png")
    pipeline()
    retrieval_chart()
    diagnosis()
    generation_chart()
    print("Figures written to", OUT)
