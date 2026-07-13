import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

TEAL = "#0F766E"
DARK = "#334155"
VERM = "#D55E00"
GREY = "#C3CBD2"
NODE = "#0F766E"
BASE = "results/gies_cache/n200_s123_p20"


def load(f):
    d = json.load(open(f))
    return [tuple(e) for e in (d.get("edges", d) if isinstance(d, dict) else d)]


real = load(f"{BASE}/real_data_edges.json")
cpa = load(f"{BASE}/cpa_real_edges.json")
R = set(real)
P = set(cpa)
# ---- FULL-network honest stats ----
sh = len(R & P)
prec = sh / len(P)
rec = sh / len(R)
f1 = 2 * prec * rec / (prec + rec)
print(
    f"FULL K562 s123: real={len(R)} cpa={len(P)} shared={sh} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f}"
)

# ---- legible EXCERPT: induce on top-K hub genes by union degree ----
U = nx.Graph()
U.add_edges_from(R | P)
K = 26
hub = sorted(U.nodes(), key=lambda g: -U.degree(g))[:K]
H = set(hub)


def within(E):
    return set((a, b) for (a, b) in E if a in H and b in H)


Rs = within(R)
Ps = within(P)
shared = Rs & Ps
spur = Ps - Rs
miss = Rs - Ps
Ug = nx.Graph()
Ug.add_nodes_from(hub)
Ug.add_edges_from(Rs | Ps)
pos = nx.kamada_kawai_layout(Ug, scale=1.0)
deg = dict(Ug.degree())
sizes = [240 + deg[g] * 34 for g in Ug.nodes()]
lblpos = {g: (x, y - 0.058) for g, (x, y) in pos.items()}

fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.6))
fig.patch.set_alpha(0)


def draw(ax, title, layers):
    for edges, color, style, w, al in layers:
        Dg = nx.DiGraph()
        Dg.add_nodes_from(Ug.nodes())
        Dg.add_edges_from(edges)
        nx.draw_networkx_edges(
            Dg,
            pos,
            ax=ax,
            edge_color=color,
            style=style,
            width=w,
            alpha=al,
            arrows=True,
            arrowsize=8,
            arrowstyle="-|>",
            node_size=sizes,
            connectionstyle="arc3,rad=0.08",
        )
    nx.draw_networkx_nodes(
        Ug,
        pos,
        ax=ax,
        node_color=NODE,
        node_size=sizes,
        edgecolors="#0B5A54",
        linewidths=0.7,
    )
    nx.draw_networkx_labels(
        Ug, lblpos, ax=ax, font_size=7, font_color="#1f2937", font_weight="bold"
    )
    ax.set_title(title, fontsize=15, fontweight="bold", color="#0f172a", pad=8)
    ax.margins(0.13)
    ax.axis("off")


draw(axes[0], "Real Perturb-seq  →  recovered network", [(Rs, DARK, "solid", 1.2, 0.9)])
axes[0].text(
    0.5,
    -0.05,
    "ground truth",
    transform=axes[0].transAxes,
    ha="center",
    fontsize=11,
    color="#475569",
)
draw(
    axes[1],
    "CPA (deep VCM)  →  recovered network",
    [
        (miss, GREY, "dashed", 1.1, 0.85),
        (shared, TEAL, "solid", 1.5, 0.95),
        (spur, VERM, "solid", 1.3, 0.95),
    ],
)
axes[1].text(
    0.5,
    -0.05,
    f"full-network edge-set F1 = {f1:.2f}  (precision {prec:.2f} · recall {rec:.2f})",
    transform=axes[1].transAxes,
    ha="center",
    fontsize=11,
    color="#475569",
)
leg = [
    Line2D([0], [0], color=TEAL, lw=2.6, label="correct edge (in real)"),
    Line2D([0], [0], color=VERM, lw=2.6, label="spurious edge (wrong)"),
    Line2D([0], [0], color=GREY, lw=2, ls="--", label="missed real edge"),
]
axes[1].legend(
    handles=leg,
    loc="lower right",
    fontsize=8.5,
    frameon=False,
    bbox_to_anchor=(1.03, -0.02),
)
fig.suptitle(
    "Matching per-gene levels ≠ getting the wiring right",
    fontsize=16.5,
    fontweight="bold",
    color="#0f172a",
    y=1.0,
)
fig.text(
    0.5,
    0.925,
    "K562 · same genes, same layout · excerpt of the top-connectivity subnetwork (full network: 200 genes, ~1,200 edges).",
    ha="center",
    fontsize=10.5,
    color="#64748b",
)
plt.tight_layout(rect=[0, 0, 1, 0.9])
out = "/Users/aayanalwani/Desktop/ISMB_Poster_PerturbCausal/figures/real_vs_predicted_grn.png"
plt.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
print("SAVED", out)
