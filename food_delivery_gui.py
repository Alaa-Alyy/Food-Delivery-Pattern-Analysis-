"""
╔══════════════════════════════════════════════════════════════════╗
║        Food Delivery Pattern Analysis — Professional GUI         ║
║  Requirements: pip install pandas numpy networkx matplotlib      ║
║               seaborn scipy mlxtend Pillow                       ║
║  Run: python food_delivery_gui.py                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os, sys, ast, time, itertools, collections, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  THEME & COLOURS
# ─────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0D0D14",
    "surface":   "#13131C",
    "card":      "#1A1A26",
    "border":    "#252535",
    "accent":    "#FF6B35",
    "accent2":   "#00D4AA",
    "accent3":   "#A78BFA",
    "yellow":    "#FFD166",
    "text":      "#F0EEE8",
    "muted":     "#7C7A90",
    "success":   "#22C55E",
    "danger":    "#EF4444",
    "white":     "#FFFFFF",
}

MPL_DARK = {
    "axes.facecolor":    C["card"],
    "figure.facecolor":  C["surface"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["text"],
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "grid.color":        C["border"],
    "text.color":        C["text"],
    "axes.titlecolor":   C["text"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        100,
}
plt.rcParams.update(MPL_DARK)

PALETTES = {
    "accent":   [C["accent"], C["accent2"], C["accent3"], C["yellow"], "#F472B6", "#60A5FA", "#34D399", "#FB923C"],
    "heatmap":  "RdYlGn",
    "gradient": "magma",
}

FONT = {
    "title":   ("Segoe UI", 20, "bold"),
    "heading": ("Segoe UI", 14, "bold"),
    "subhead": ("Segoe UI", 11, "bold"),
    "body":    ("Segoe UI", 10),
    "small":   ("Segoe UI", 9),
    "mono":    ("Consolas", 9),
    "big":     ("Segoe UI", 32, "bold"),
    "hero":    ("Segoe UI", 28, "bold"),
}

# ─────────────────────────────────────────────────────────────────
#  DATA PATHS  (auto-resolve next to this script)
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ORDERS_CSV  = SCRIPT_DIR / "data" / "data" / "final_dataset2.csv"
REVIEWS_CSV = SCRIPT_DIR / "data" / "data" / "ULTRA_FINAL_REVIEWS.csv"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"


# ══════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════
class DataStore:
    """Loads and caches all project data."""

    def __init__(self):
        self.orders_df   = None
        self.reviews_df  = None
        self.encoded_df  = None
        self.transactions = None
        self.item_freq   = None
        self.co_occur    = None
        self.graph       = None
        self.pagerank    = None
        self._loaded     = False

    # ── loaders ──────────────────────────────────────────────────
    def load(self, orders_path=None, reviews_path=None):
        op = Path(orders_path)  if orders_path  else ORDERS_CSV
        rp = Path(reviews_path) if reviews_path else REVIEWS_CSV

        # Orders
        self.orders_df = pd.read_csv(op, engine="python", on_bad_lines="skip")
        self.orders_df["Items_Parsed"] = self.orders_df["Items"].apply(
            lambda x: [i.strip() for i in str(x).split(",") if i.strip()]
        )

        # Transactions for Apriori / FP-Growth
        try:
            self.transactions = self.orders_df["Items_List"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            ).tolist()
        except Exception:
            self.transactions = self.orders_df["Items_Parsed"].tolist()

        # Item frequencies
        all_items = [i for sub in self.orders_df["Items_Parsed"] for i in sub]
        self.item_freq = collections.Counter(all_items)

        # Build co-occurrence + graph
        multi = self.orders_df[self.orders_df["Items_Parsed"].apply(len) >= 2]["Items_Parsed"].tolist()
        self.co_occur = collections.Counter()
        for t in multi:
            for a, b in itertools.combinations(sorted(t), 2):
                self.co_occur[(a, b)] += 1

        self.graph = nx.Graph()
        for (a, b), w in self.co_occur.items():
            self.graph.add_edge(a, b, weight=w)

        self.pagerank = nx.pagerank(self.graph, alpha=0.85, weight="weight")

        # Reviews
        if rp.exists():
            self.reviews_df = pd.read_csv(rp, engine="python", on_bad_lines="skip")
        else:
            self.reviews_df = None

        self._loaded = True

    # ── helpers ──────────────────────────────────────────────────
    def one_hot(self):
        """One-hot encode transactions (needed for Apriori/FP-Growth)."""
        if self.encoded_df is not None:
            return self.encoded_df
        all_items = sorted({i for t in self.transactions for i in t})
        rows = []
        for t in self.transactions:
            tset = set(t)
            rows.append({item: (item in tset) for item in all_items})
        self.encoded_df = pd.DataFrame(rows)
        return self.encoded_df

    def apriori_manual(self, min_support=0.05):
        """Pure-Python Apriori (no mlxtend needed)."""
        n = len(self.transactions)
        min_count = min_support * n

        # 1-itemsets
        freq1 = {frozenset([item]): cnt
                 for item, cnt in self.item_freq.items()
                 if cnt >= min_count}

        all_freq = dict(freq1)
        prev = list(freq1.keys())
        k = 2

        while prev:
            candidates = {}
            items = sorted({i for fs in prev for i in fs})
            for a, b in itertools.combinations(items, 1 if k == 1 else k):
                c = frozenset(list(a) if k == 1 else [a, b])
                if len(c) == k:
                    candidates[c] = 0

            # Count
            for t in self.transactions:
                tset = frozenset(t)
                for c in candidates:
                    if c.issubset(tset):
                        candidates[c] += 1

            freq_k = {c: cnt for c, cnt in candidates.items() if cnt >= min_count}
            all_freq.update(freq_k)
            prev = list(freq_k.keys())
            k += 1
            if k > 4:
                break

        result = [{"itemsets": fs, "support": cnt / n}
                  for fs, cnt in all_freq.items()]
        return pd.DataFrame(result)

    def association_rules_manual(self, freq_df, min_confidence=0.4, min_lift=1.0):
        """Derive association rules from frequent itemsets."""
        n = len(self.transactions)
        item_support = {
            frozenset([item]): cnt / n
            for item, cnt in self.item_freq.items()
        }
        for _, row in freq_df.iterrows():
            item_support[row["itemsets"]] = row["support"]

        rules = []
        for _, row in freq_df.iterrows():
            fs = row["itemsets"]
            if len(fs) < 2:
                continue
            for i in range(1, len(fs)):
                for ant in itertools.combinations(sorted(fs), i):
                    ant_fs = frozenset(ant)
                    cons_fs = fs - ant_fs
                    if not cons_fs:
                        continue
                    ant_sup  = item_support.get(ant_fs, 0)
                    cons_sup = item_support.get(cons_fs, 0)
                    rule_sup = row["support"]
                    if ant_sup == 0 or cons_sup == 0:
                        continue
                    conf = rule_sup / ant_sup
                    lift = conf / cons_sup
                    if conf >= min_confidence and lift >= min_lift:
                        rules.append({
                            "antecedents":  ant_fs,
                            "consequents":  cons_fs,
                            "support":      rule_sup,
                            "confidence":   conf,
                            "lift":         lift,
                        })
        return pd.DataFrame(rules)

    def try_mlxtend(self, min_support=0.05):
        """Use mlxtend if available, else fall back to manual."""
        try:
            from mlxtend.frequent_patterns import apriori as mxt_apriori, fpgrowth, association_rules
            from mlxtend.preprocessing import TransactionEncoder
            te = TransactionEncoder()
            ary = te.fit(self.transactions).transform(self.transactions)
            enc = pd.DataFrame(ary, columns=te.columns_)
            freq = mxt_apriori(enc, min_support=min_support, use_colnames=True)
            freq_fp = fpgrowth(enc, min_support=min_support, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1.0)
            strong = rules[(rules["confidence"] > 0.4) & (rules["lift"] > 1.0)]
            return freq, freq_fp, strong, True
        except ImportError:
            freq = self.apriori_manual(min_support)
            rules = self.association_rules_manual(freq)
            return freq, freq.copy(), rules, False


DS = DataStore()


# ══════════════════════════════════════════════════════════════════
#  WIDGET HELPERS
# ══════════════════════════════════════════════════════════════════
def styled_frame(parent, bg=None, **kw):
    return tk.Frame(parent, bg=bg or C["bg"], **kw)

def label(parent, text, font=None, fg=None, bg=None, **kw):
    return tk.Label(parent, text=text,
                    font=font or FONT["body"],
                    fg=fg or C["text"],
                    bg=bg or C["bg"], **kw)

def stat_card(parent, title, value, subtitle="", color=None):
    color = color or C["accent"]
    frame = tk.Frame(parent, bg=C["card"],
                     highlightbackground=C["border"], highlightthickness=1)

    # colour bar on left
    bar = tk.Frame(frame, bg=color, width=4)
    bar.pack(side="left", fill="y")

    inner = tk.Frame(frame, bg=C["card"], padx=16, pady=14)
    inner.pack(side="left", fill="both", expand=True)

    tk.Label(inner, text=title.upper(), font=("Segoe UI", 8, "bold"),
             fg=C["muted"], bg=C["card"], anchor="w").pack(anchor="w")
    tk.Label(inner, text=value, font=FONT["big"],
             fg=color, bg=C["card"], anchor="w").pack(anchor="w")
    if subtitle:
        tk.Label(inner, text=subtitle, font=FONT["small"],
                 fg=C["muted"], bg=C["card"], anchor="w").pack(anchor="w")
    return frame

def section_title(parent, text, bg=None):
    f = tk.Frame(parent, bg=bg or C["bg"])
    tk.Label(f, text=text, font=FONT["heading"],
             fg=C["text"], bg=bg or C["bg"]).pack(side="left")
    tk.Frame(f, bg=C["border"], height=1).pack(side="left", fill="x",
                                                expand=True, padx=(12, 0))
    return f

def pill(parent, text, color=None, bg=None):
    color = color or C["accent"]
    bg    = bg    or C["card"]
    return tk.Label(parent, text=f"  {text}  ",
                    font=("Segoe UI", 9, "bold"),
                    fg=color, bg=bg,
                    relief="flat", bd=0,
                    highlightbackground=color, highlightthickness=1)

def scrolled_frame(parent, bg=None):
    bg = bg or C["bg"]
    canvas = tk.Canvas(parent, bg=bg, highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    sf = tk.Frame(canvas, bg=bg)
    sf.bind("<Configure>", lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=sf, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(e):
        canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    return sf

def embed_figure(parent, fig, toolbar=True):
    """Embed a matplotlib figure in a tkinter frame."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(fill="both", expand=True)
    if toolbar:
        tb_frame = tk.Frame(parent, bg=C["surface"])
        tb_frame.pack(fill="x")
        tb = NavigationToolbar2Tk(canvas, tb_frame)
        tb.config(background=C["surface"])
        for child in tb.winfo_children():
            try:
                child.config(background=C["surface"], foreground=C["text"])
            except Exception:
                pass
        tb.update()
    return canvas

def loading_overlay(parent, text="Analysing…"):
    ov = tk.Frame(parent, bg=C["bg"])
    ov.place(relwidth=1, relheight=1)
    tk.Label(ov, text="⏳", font=("Segoe UI", 48), bg=C["bg"]).pack(expand=True)
    tk.Label(ov, text=text, font=FONT["heading"], fg=C["muted"], bg=C["bg"]).pack()
    return ov


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR NAV
# ══════════════════════════════════════════════════════════════════
class Sidebar(tk.Frame):
    ITEMS = [
        ("🏠",  "Overview",      "overview"),
        ("📊",  "Dataset",       "dataset"),
        ("🔗",  "Association\nRules", "apriori"),
        ("🌐",  "PageRank",      "pagerank"),
        ("🤖",  "BERT Sentiment","bert"),
        ("📝",  "Reviews",       "reviews"),
        ("⚙️",  "Settings",      "settings"),
    ]

    def __init__(self, parent, on_nav):
        super().__init__(parent, bg=C["surface"], width=210)
        self.pack_propagate(False)
        self.on_nav    = on_nav
        self._buttons  = {}
        self._active   = None
        self._build()

    def _build(self):
        # ── Logo ──────────────────────────────────────────────────
        logo_f = tk.Frame(self, bg=C["surface"], pady=24, padx=20)
        logo_f.pack(fill="x")
        tk.Label(logo_f, text="🍔", font=("Segoe UI", 30),
                 bg=C["surface"]).pack()
        tk.Label(logo_f, text="Food Delivery",
                 font=("Segoe UI", 13, "bold"),
                 fg=C["accent"], bg=C["surface"]).pack()
        tk.Label(logo_f, text="Pattern Analysis",
                 font=("Segoe UI", 10),
                 fg=C["muted"], bg=C["surface"]).pack()

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=16)

        # ── Nav items ─────────────────────────────────────────────
        nav_f = tk.Frame(self, bg=C["surface"], pady=12)
        nav_f.pack(fill="x")

        for icon, text, key in self.ITEMS:
            btn = self._nav_btn(nav_f, icon, text, key)
            btn.pack(fill="x", padx=12, pady=2)
            self._buttons[key] = btn

        # ── Footer badge ──────────────────────────────────────────
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=16, side="bottom", pady=(0, 8))
        footer = tk.Frame(self, bg=C["surface"])
        footer.pack(side="bottom", fill="x", padx=16, pady=10)
        dot = tk.Label(footer, text="●", font=("Segoe UI", 8),
                       fg=C["success"], bg=C["surface"])
        dot.pack(side="left")
        tk.Label(footer, text=" 20,000 Orders", font=FONT["small"],
                 fg=C["muted"], bg=C["surface"]).pack(side="left")

        self.set_active("overview")

    def _nav_btn(self, parent, icon, text, key):
        f = tk.Frame(parent, bg=C["surface"], cursor="hand2",
                     pady=10, padx=12)
        tk.Label(f, text=icon, font=("Segoe UI", 14),
                 bg=C["surface"], fg=C["text"], width=2).pack(side="left")
        tk.Label(f, text=text, font=("Segoe UI", 10),
                 bg=C["surface"], fg=C["muted"],
                 justify="left", anchor="w").pack(side="left", padx=8)

        def enter(_): 
            if key != self._active:
                f.config(bg=C["card"])
                for w in f.winfo_children(): w.config(bg=C["card"])
        def leave(_):
            if key != self._active:
                f.config(bg=C["surface"])
                for w in f.winfo_children(): w.config(bg=C["surface"])
        def click(_): self.set_active(key); self.on_nav(key)

        f.bind("<Enter>",   enter)
        f.bind("<Leave>",   leave)
        f.bind("<Button-1>", click)
        for w in f.winfo_children():
            w.bind("<Enter>",   enter)
            w.bind("<Leave>",   leave)
            w.bind("<Button-1>", click)
        return f

    def set_active(self, key):
        if self._active and self._active in self._buttons:
            old = self._buttons[self._active]
            old.config(bg=C["surface"])
            for w in old.winfo_children():
                w.config(bg=C["surface"])
                if isinstance(w, tk.Label):
                    w.config(fg=C["muted"])

        self._active = key
        if key in self._buttons:
            btn = self._buttons[key]
            btn.config(bg=C["card"],
                       highlightbackground=C["accent"],
                       highlightthickness=1)
            for w in btn.winfo_children():
                w.config(bg=C["card"])
                if isinstance(w, tk.Label):
                    w.config(fg=C["text"])


# ══════════════════════════════════════════════════════════════════
#  TOP BAR
# ══════════════════════════════════════════════════════════════════
class Topbar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["surface"], height=56,
                         highlightbackground=C["border"], highlightthickness=1)
        self.pack_propagate(False)
        self._title_var = tk.StringVar(value="Overview")
        self._status_var = tk.StringVar(value="Ready")

        left = tk.Frame(self, bg=C["surface"])
        left.pack(side="left", padx=20, fill="y")
        tk.Label(left, textvariable=self._title_var,
                 font=FONT["heading"], fg=C["text"],
                 bg=C["surface"]).pack(side="left", pady=14)

        right = tk.Frame(self, bg=C["surface"])
        right.pack(side="right", padx=20, fill="y")

        self._status_lbl = tk.Label(right, textvariable=self._status_var,
                                    font=FONT["small"], fg=C["muted"],
                                    bg=C["surface"])
        self._status_lbl.pack(side="right", pady=14, padx=(8, 0))

        for tag, col in [("Talabat / Uber Eats", C["muted"]),
                          ("Egypt Market", C["muted"])]:
            tk.Label(right, text=f"  {tag}  ", font=("Segoe UI", 9),
                     fg=col, bg=C["border"],
                     relief="flat").pack(side="right", padx=4)

    def set_title(self, t):  self._title_var.set(t)
    def set_status(self, s): self._status_var.set(s)


# ══════════════════════════════════════════════════════════════════
#  PAGE BASE
# ══════════════════════════════════════════════════════════════════
class BasePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._built = False

    def activate(self):
        if not self._built:
            self._build()
            self._built = True
        self.tkraise()

    def _build(self): pass


# ══════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════
class OverviewPage(BasePage):
    def _build(self):
        sf = scrolled_frame(self)

        # ── Hero ──────────────────────────────────────────────────
        hero = tk.Frame(sf, bg=C["card"],
                        highlightbackground=C["border"], highlightthickness=1,
                        pady=30, padx=36)
        hero.pack(fill="x", padx=20, pady=(20, 0))

        tk.Label(hero, text="📦  FOOD DELIVERY · PATTERN ANALYSIS",
                 font=("Segoe UI", 9, "bold"),
                 fg=C["accent"], bg=C["card"]).pack(anchor="w")
        tk.Label(hero, text="Behaviour Insights\nfrom 20,000 Orders",
                 font=FONT["hero"], fg=C["text"], bg=C["card"],
                 justify="left").pack(anchor="w", pady=(8, 10))
        tk.Label(hero,
                 text="Exploring customer ordering patterns across 10 restaurants using\n"
                      "Apriori, FP-Growth, PageRank and BERT sentiment analysis.",
                 font=FONT["body"], fg=C["muted"], bg=C["card"],
                 justify="left").pack(anchor="w")

        # ── Stat Cards ────────────────────────────────────────────
        row = tk.Frame(sf, bg=C["bg"])
        row.pack(fill="x", padx=20, pady=14)

        cards = [
            ("Total Orders",   "20K",  "Synthetic transactions",  C["accent"]),
            ("Restaurants",    "10",   "McDonalds, KFC, Zooba…",  C["accent2"]),
            ("User Types",     "3",    "Student · Family · Diet", C["accent3"]),
            ("ML Methods",     "4",    "Apriori FP-Growth PR BERT", C["yellow"]),
        ]
        for i, (t, v, s, c) in enumerate(cards):
            row.columnconfigure(i, weight=1)
            sc = stat_card(row, t, v, s, c)
            sc.grid(row=0, column=i, sticky="nsew", padx=6)

        # ── Method Cards ──────────────────────────────────────────
        section_title(sf, "  Analysis Methods").pack(fill="x",
                                                      padx=20, pady=(16, 8))
        methods_f = tk.Frame(sf, bg=C["bg"])
        methods_f.pack(fill="x", padx=20)

        methods = [
            ("🔗", "Apriori",    C["accent"],
             "Frequent itemset mining. Discovers which menu items appear\n"
             "together most often in the same order."),
            ("⚡", "FP-Growth",  C["accent2"],
             "Fast pattern mining using a compressed FP-tree structure.\n"
             "Faster than Apriori for large datasets."),
            ("🌐", "PageRank",   C["accent3"],
             "Graph-based ranking of food items by influence.\n"
             "Items co-ordered often gain higher centrality scores."),
            ("🤖", "BERT",       C["yellow"],
             "Transformer-based sentiment analysis on customer reviews.\n"
             "Classifies each review as Positive or Negative."),
        ]
        for col, (icon, name, color, desc) in enumerate(methods):
            methods_f.columnconfigure(col, weight=1)
            card = tk.Frame(methods_f, bg=C["card"],
                            highlightbackground=C["border"], highlightthickness=1,
                            padx=18, pady=18)
            card.grid(row=0, column=col, sticky="nsew", padx=6)
            tk.Label(card, text=icon, font=("Segoe UI", 24),
                     bg=C["card"], fg=color).pack(anchor="w")
            tk.Label(card, text=name, font=FONT["subhead"],
                     fg=color, bg=C["card"]).pack(anchor="w", pady=(4, 0))
            tk.Label(card, text=desc, font=FONT["small"],
                     fg=C["muted"], bg=C["card"],
                     justify="left", wraplength=160).pack(anchor="w", pady=(6, 0))

        # ── Output images preview ─────────────────────────────────
        if OUTPUTS_DIR.exists():
            pngs = sorted(OUTPUTS_DIR.glob("*.png"))
            if pngs:
                section_title(sf, "  Saved Outputs").pack(
                    fill="x", padx=20, pady=(20, 8))
                imgs_f = tk.Frame(sf, bg=C["bg"])
                imgs_f.pack(fill="x", padx=20, pady=(0, 20))

                from PIL import Image, ImageTk
                cols = 4
                for idx, p in enumerate(pngs[:8]):
                    try:
                        img = Image.open(p)
                        img.thumbnail((240, 160))
                        photo = ImageTk.PhotoImage(img)
                        imgs_f.columnconfigure(idx % cols, weight=1)
                        c_frame = tk.Frame(imgs_f, bg=C["card"],
                                           highlightbackground=C["border"],
                                           highlightthickness=1, padx=4, pady=4)
                        c_frame.grid(row=idx // cols, column=idx % cols,
                                     sticky="nsew", padx=5, pady=5)
                        lbl = tk.Label(c_frame, image=photo, bg=C["card"])
                        lbl.image = photo
                        lbl.pack()
                        tk.Label(c_frame, text=p.stem,
                                 font=FONT["small"], fg=C["muted"],
                                 bg=C["card"]).pack()
                    except Exception:
                        pass


# ══════════════════════════════════════════════════════════════════
#  PAGE: DATASET
# ══════════════════════════════════════════════════════════════════
class DatasetPage(BasePage):
    def _build(self):
        sf = scrolled_frame(self)

        # ── Stats row ─────────────────────────────────────────────
        row = tk.Frame(sf, bg=C["bg"])
        row.pack(fill="x", padx=20, pady=(20, 0))

        rc  = DS.orders_df["Restaurant"].value_counts()
        utc = DS.orders_df["User_Type"].value_counts()
        tc  = DS.orders_df["Time"].value_counts()

        cards = [
            ("Total Rows",     f"{len(DS.orders_df):,}",      "",       C["accent"]),
            ("Top Restaurant", rc.index[0],                   f"{rc.iloc[0]:,} orders", C["accent2"]),
            ("Top User Type",  utc.index[0],                  f"{utc.iloc[0]:,} users", C["accent3"]),
            ("Peak Time",      tc.index[0].title(),           f"{tc.iloc[0]:,} orders", C["yellow"]),
        ]
        for i, (t, v, s, c) in enumerate(cards):
            row.columnconfigure(i, weight=1)
            stat_card(row, t, v, s, c).grid(row=0, column=i,
                                             sticky="nsew", padx=6)

        # ── Restaurant distribution ────────────────────────────────
        section_title(sf, "  Restaurant Order Distribution").pack(
            fill="x", padx=20, pady=(20, 8))

        fig1, ax1 = plt.subplots(figsize=(13, 3.8))
        rdata = rc.sort_values(ascending=True)
        colors = [C["accent"] if v == rdata.max() else C["accent2"]
                  for v in rdata.values]
        bars = ax1.barh(rdata.index, rdata.values, color=colors, height=0.65)
        ax1.bar_label(bars, fmt="%,.0f", padding=4,
                      color=C["muted"], fontsize=8)
        ax1.set_xlabel("Number of Orders", color=C["muted"])
        ax1.set_title("Orders per Restaurant", color=C["text"], fontsize=12, pad=12)
        fig1.tight_layout(pad=1)

        chart_f1 = tk.Frame(sf, bg=C["surface"],
                            highlightbackground=C["border"], highlightthickness=1)
        chart_f1.pack(fill="x", padx=20, pady=(0, 16))
        embed_figure(chart_f1, fig1)
        plt.close(fig1)

        # ── Pie breakdown ─────────────────────────────────────────
        row2 = tk.Frame(sf, bg=C["bg"])
        row2.pack(fill="x", padx=20, pady=(0, 16))
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        for col_i, (df_col, title) in enumerate([
                ("User_Type", "User Type Distribution"),
                ("Time",      "Order Time Distribution"),
        ]):
            fig, ax = plt.subplots(figsize=(5, 3.6))
            vals  = DS.orders_df[df_col].value_counts()
            wedge_cols = PALETTES["accent"][:len(vals)]
            wedges, texts, autotexts = ax.pie(
                vals.values, labels=vals.index,
                colors=wedge_cols, autopct="%1.1f%%",
                startangle=120, pctdistance=0.78,
                wedgeprops={"linewidth": 2, "edgecolor": C["surface"]})
            for at in autotexts:
                at.set_color(C["bg"]); at.set_fontsize(8)
            for t in texts:
                t.set_color(C["muted"]); t.set_fontsize(9)
            ax.set_title(title, color=C["text"], fontsize=11, pad=8)
            fig.tight_layout()

            cf = tk.Frame(row2, bg=C["surface"],
                          highlightbackground=C["border"], highlightthickness=1)
            cf.grid(row=0, column=col_i, sticky="nsew", padx=6)
            embed_figure(cf, fig)
            plt.close(fig)

        # ── Top items bar ─────────────────────────────────────────
        section_title(sf, "  Top 20 Most Ordered Items").pack(
            fill="x", padx=20, pady=(4, 8))

        fig2, ax2 = plt.subplots(figsize=(13, 4.5))
        top20 = DS.item_freq.most_common(20)
        names, cnts = zip(*top20)
        bar_colors = [C["accent"] if i < 3 else C["accent2"]
                      if i < 8 else C["border"] for i in range(len(names))]
        bars2 = ax2.bar(names, cnts, color=bar_colors, width=0.7)
        ax2.bar_label(bars2, fmt="%,.0f", color=C["muted"], fontsize=7.5)
        ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("Frequency", color=C["muted"])
        ax2.set_title("Top 20 Most Ordered Items", color=C["text"],
                      fontsize=12, pad=12)
        fig2.tight_layout(pad=1)

        cf2 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf2.pack(fill="x", padx=20, pady=(0, 20))
        embed_figure(cf2, fig2)
        plt.close(fig2)

        # ── Data table preview ────────────────────────────────────
        section_title(sf, "  Data Preview (first 200 rows)").pack(
            fill="x", padx=20, pady=(4, 8))

        tbl_f = tk.Frame(sf, bg=C["surface"],
                         highlightbackground=C["border"], highlightthickness=1)
        tbl_f.pack(fill="x", padx=20, pady=(0, 20))

        cols = ["Order_ID", "Restaurant", "User_Type", "Time", "Items"]
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background=C["card"], foreground=C["text"],
                        fieldbackground=C["card"], borderwidth=0,
                        rowheight=24)
        style.configure("Dark.Treeview.Heading",
                        background=C["surface"], foreground=C["muted"],
                        relief="flat", borderwidth=0)
        style.map("Dark.Treeview",
                  background=[("selected", C["accent"])],
                  foreground=[("selected", C["white"])])

        tv = ttk.Treeview(tbl_f, columns=cols, show="headings",
                          style="Dark.Treeview", height=12)
        sb = ttk.Scrollbar(tbl_f, orient="vertical", command=tv.yview)
        tv.configure(yscroll=sb.set)

        widths = [70, 140, 90, 90, 400]
        for c, w in zip(cols, widths):
            tv.heading(c, text=c)
            tv.column(c, width=w, anchor="w")

        for _, row in DS.orders_df[cols].head(200).iterrows():
            tv.insert("", "end", values=list(row))

        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")


# ══════════════════════════════════════════════════════════════════
#  PAGE: ASSOCIATION RULES (Apriori + FP-Growth)
# ══════════════════════════════════════════════════════════════════
class AprioriPage(BasePage):
    def _build(self):
        # controls
        ctrl = tk.Frame(self, bg=C["surface"],
                        highlightbackground=C["border"], highlightthickness=1,
                        pady=10, padx=16)
        ctrl.pack(fill="x")

        tk.Label(ctrl, text="Min Support:", font=FONT["small"],
                 fg=C["muted"], bg=C["surface"]).pack(side="left")
        self._sup_var = tk.DoubleVar(value=0.05)
        sup_scale = tk.Scale(ctrl, from_=0.01, to=0.3, resolution=0.01,
                             variable=self._sup_var, orient="horizontal",
                             length=200, bg=C["surface"], fg=C["text"],
                             highlightthickness=0, troughcolor=C["border"],
                             activebackground=C["accent"], sliderrelief="flat")
        sup_scale.pack(side="left", padx=8)

        tk.Label(ctrl, text="Min Confidence:", font=FONT["small"],
                 fg=C["muted"], bg=C["surface"]).pack(side="left", padx=(16, 0))
        self._conf_var = tk.DoubleVar(value=0.4)
        conf_scale = tk.Scale(ctrl, from_=0.1, to=0.9, resolution=0.05,
                              variable=self._conf_var, orient="horizontal",
                              length=200, bg=C["surface"], fg=C["text"],
                              highlightthickness=0, troughcolor=C["border"],
                              activebackground=C["accent2"], sliderrelief="flat")
        conf_scale.pack(side="left", padx=8)

        run_btn = tk.Button(ctrl, text="  ▶  Run Analysis  ",
                            font=FONT["subhead"], fg=C["white"],
                            bg=C["accent"], activebackground="#e55a28",
                            relief="flat", cursor="hand2",
                            command=self._run)
        run_btn.pack(side="left", padx=16)

        self._status = tk.Label(ctrl, text="", font=FONT["small"],
                                fg=C["muted"], bg=C["surface"])
        self._status.pack(side="left")

        # content area
        self._content = tk.Frame(self, bg=C["bg"])
        self._content.pack(fill="both", expand=True)

        # run on load
        self._run()

    def _run(self):
        for w in self._content.winfo_children():
            w.destroy()

        ov = loading_overlay(self._content, "Mining association rules…")
        self.update()

        def work():
            sup  = self._sup_var.get()
            conf = self._conf_var.get()

            t0 = time.time()
            freq, freq_fp, rules, has_mlx = DS.try_mlxtend(min_support=sup)
            elapsed = time.time() - t0

            self.after(0, lambda: self._render(freq, freq_fp, rules,
                                               has_mlx, elapsed, sup, conf, ov))

        threading.Thread(target=work, daemon=True).start()

    def _render(self, freq, freq_fp, rules, has_mlx, elapsed, sup, conf, ov):
        ov.destroy()
        sf = scrolled_frame(self._content)

        engine = "mlxtend" if has_mlx else "built-in (install mlxtend for full features)"
        self._status.config(
            text=f"Engine: {engine} | Items: {len(freq)} | Rules: {len(rules)} | {elapsed:.2f}s"
        )

        # stat row
        row = tk.Frame(sf, bg=C["bg"])
        row.pack(fill="x", padx=20, pady=(20, 0))
        cards = [
            ("Frequent Itemsets",  str(len(freq)),                 f"support ≥ {sup}",       C["accent"]),
            ("Strong Rules",       str(len(rules)),                f"conf ≥ {conf}",          C["accent2"]),
            ("Unique Items",       str(len(DS.item_freq)),         "in dataset",              C["accent3"]),
            ("Top Rule Lift",
             f"{rules['lift'].max():.2f}" if len(rules) else "—",  "max lift found",         C["yellow"]),
        ]
        for i, (t, v, s, c) in enumerate(cards):
            row.columnconfigure(i, weight=1)
            stat_card(row, t, v, s, c).grid(row=0, column=i, sticky="nsew", padx=6)

        # ── Top Frequent Itemsets bar ─────────────────────────────
        section_title(sf, "  Top 15 Frequent Itemsets").pack(
            fill="x", padx=20, pady=(20, 8))

        fig1, ax1 = plt.subplots(figsize=(13, 4))
        top = freq.sort_values("support", ascending=False).head(15).copy()
        top["label"] = top["itemsets"].apply(lambda x: ", ".join(sorted(x)))
        top = top.sort_values("support")
        bar_c = plt.cm.get_cmap("plasma")(np.linspace(0.3, 0.9, len(top)))
        bars = ax1.barh(top["label"], top["support"] * 100,
                        color=bar_c, height=0.65)
        ax1.bar_label(bars, fmt="%.1f%%", padding=4,
                      color=C["muted"], fontsize=8)
        ax1.set_xlabel("Support %", color=C["muted"])
        ax1.set_title("Top 15 Frequent Itemsets", color=C["text"],
                      fontsize=12, pad=10)
        fig1.tight_layout(pad=1)

        cf1 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf1.pack(fill="x", padx=20, pady=(0, 16))
        embed_figure(cf1, fig1)
        plt.close(fig1)

        if len(rules) == 0:
            tk.Label(sf, text="No rules found with current thresholds. "
                              "Try lowering support or confidence.",
                     font=FONT["body"], fg=C["muted"], bg=C["bg"]).pack(pady=20)
            return

        # ── Support vs Confidence scatter ─────────────────────────
        section_title(sf, "  Support vs Confidence (all rules)").pack(
            fill="x", padx=20, pady=(4, 8))

        fig2, ax2 = plt.subplots(figsize=(13, 4.5))
        sc = ax2.scatter(rules["support"], rules["confidence"],
                         c=rules["lift"], cmap="plasma",
                         s=60, alpha=0.75, edgecolors="none")
        plt.colorbar(sc, ax=ax2, label="Lift", shrink=0.8)
        ax2.set_xlabel("Support", color=C["muted"])
        ax2.set_ylabel("Confidence", color=C["muted"])
        ax2.set_title("Support vs Confidence coloured by Lift",
                      color=C["text"], fontsize=12, pad=10)
        fig2.tight_layout(pad=1)

        cf2 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf2.pack(fill="x", padx=20, pady=(0, 16))
        embed_figure(cf2, fig2)
        plt.close(fig2)

        # ── Rules Table ───────────────────────────────────────────
        section_title(sf, "  Top 30 Association Rules").pack(
            fill="x", padx=20, pady=(4, 8))

        tbl_f = tk.Frame(sf, bg=C["surface"],
                         highlightbackground=C["border"], highlightthickness=1)
        tbl_f.pack(fill="x", padx=20, pady=(0, 20))

        cols = ["Antecedent", "→  Consequent", "Support", "Confidence", "Lift"]
        tv = ttk.Treeview(tbl_f, columns=cols, show="headings",
                          style="Dark.Treeview", height=12)
        sb2 = ttk.Scrollbar(tbl_f, orient="vertical", command=tv.yview)
        tv.configure(yscroll=sb2.set)

        tv.heading(cols[0], text="Antecedent")
        tv.heading(cols[1], text="→  Consequent")
        tv.heading("Support",    text="Support")
        tv.heading("Confidence", text="Confidence")
        tv.heading("Lift",       text="Lift")
        tv.column(cols[0], width=220)
        tv.column(cols[1], width=220)
        tv.column("Support",    width=90, anchor="center")
        tv.column("Confidence", width=100, anchor="center")
        tv.column("Lift",       width=80,  anchor="center")

        top_rules = rules.sort_values("confidence", ascending=False).head(30)
        for _, r in top_rules.iterrows():
            ant = ", ".join(sorted(r["antecedents"]))
            con = ", ".join(sorted(r["consequents"]))
            tv.insert("", "end", values=[ant, con,
                                          f"{r['support']:.4f}",
                                          f"{r['confidence']:.4f}",
                                          f"{r['lift']:.3f}"])

        tv.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")

        # ── Network graph ─────────────────────────────────────────
        section_title(sf, "  Rules Network Graph (top 30 rules)").pack(
            fill="x", padx=20, pady=(8, 8))

        fig3, ax3 = plt.subplots(figsize=(13, 7))
        G = nx.DiGraph()
        sub = top_rules.head(30)
        for _, r in sub.iterrows():
            ant = ", ".join(sorted(r["antecedents"]))
            con = ", ".join(sorted(r["consequents"]))
            G.add_edge(ant, con, weight=r["confidence"])

        pos = nx.spring_layout(G, seed=42, k=2.5)
        edges = G.edges(data=True)
        widths = [d["weight"] * 4 for _, _, d in edges]
        nx.draw_networkx_nodes(G, pos, node_color=C["accent"],
                               node_size=1800, alpha=0.85, ax=ax3)
        nx.draw_networkx_labels(G, pos, font_size=7, font_color=C["bg"],
                                font_weight="bold", ax=ax3)
        nx.draw_networkx_edges(G, pos, edge_color=C["accent2"],
                               width=widths, alpha=0.6,
                               arrows=True, arrowsize=18,
                               connectionstyle="arc3,rad=0.1", ax=ax3)
        ax3.set_title("Association Rules Network", color=C["text"],
                      fontsize=12, pad=10)
        ax3.axis("off")
        fig3.tight_layout()

        cf3 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf3.pack(fill="x", padx=20, pady=(0, 20))
        embed_figure(cf3, fig3)
        plt.close(fig3)

        # ── Heatmap ───────────────────────────────────────────────
        section_title(sf, "  Confidence Heatmap").pack(
            fill="x", padx=20, pady=(4, 8))

        try:
            pivot_rules = top_rules.copy()
            pivot_rules["ant_str"] = pivot_rules["antecedents"].apply(
                lambda x: ", ".join(sorted(x)))
            pivot_rules["con_str"] = pivot_rules["consequents"].apply(
                lambda x: ", ".join(sorted(x)))
            heat_data = pivot_rules.pivot_table(
                index="ant_str", columns="con_str",
                values="confidence", aggfunc="max")

            fig4, ax4 = plt.subplots(figsize=(13, max(4, len(heat_data) * 0.45)))
            sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="YlOrRd",
                        ax=ax4, linewidths=0.3,
                        linecolor=C["border"],
                        cbar_kws={"shrink": 0.8})
            ax4.set_title("Confidence Heatmap", color=C["text"],
                          fontsize=12, pad=10)
            ax4.tick_params(axis="x", rotation=35, labelsize=7.5)
            ax4.tick_params(axis="y", rotation=0, labelsize=7.5)
            fig4.tight_layout(pad=1)

            cf4 = tk.Frame(sf, bg=C["surface"],
                           highlightbackground=C["border"], highlightthickness=1)
            cf4.pack(fill="x", padx=20, pady=(0, 20))
            embed_figure(cf4, fig4)
            plt.close(fig4)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
#  PAGE: PAGERANK
# ══════════════════════════════════════════════════════════════════
class PageRankPage(BasePage):
    def _build(self):
        sf = scrolled_frame(self)

        pr_df = pd.DataFrame(list(DS.pagerank.items()),
                             columns=["Meal", "PageRank_Score"])
        pr_df = pr_df.sort_values("PageRank_Score", ascending=False).reset_index(drop=True)

        # stats
        row = tk.Frame(sf, bg=C["bg"])
        row.pack(fill="x", padx=20, pady=(20, 0))
        cards = [
            ("Graph Nodes",  f"{DS.graph.number_of_nodes()}",   "unique menu items",         C["accent"]),
            ("Graph Edges",  f"{DS.graph.number_of_edges()}",   "co-occurrence pairs",        C["accent2"]),
            ("Top Item",     pr_df.iloc[0]["Meal"],             f"PageRank {pr_df.iloc[0]['PageRank_Score']:.5f}", C["accent3"]),
            ("Density",      f"{nx.density(DS.graph):.4f}",    "graph density",              C["yellow"]),
        ]
        for i, (t, v, s, c) in enumerate(cards):
            row.columnconfigure(i, weight=1)
            stat_card(row, t, v, s, c).grid(row=0, column=i, sticky="nsew", padx=6)

        # ── PageRank bar ──────────────────────────────────────────
        section_title(sf, "  Top 15 Meals by PageRank Score").pack(
            fill="x", padx=20, pady=(20, 8))

        top15 = pr_df.head(15).sort_values("PageRank_Score")
        fig1, ax1 = plt.subplots(figsize=(13, 4.5))
        bar_c = plt.cm.get_cmap("magma")(np.linspace(0.25, 0.85, len(top15)))
        bars = ax1.barh(top15["Meal"], top15["PageRank_Score"],
                        color=bar_c, height=0.65)
        ax1.bar_label(bars, fmt="%.5f", padding=4,
                      color=C["muted"], fontsize=7.5)
        ax1.set_xlabel("PageRank Score", color=C["muted"])
        ax1.set_title("Meal Influence Ranking (PageRank α=0.85)",
                      color=C["text"], fontsize=12, pad=10)
        fig1.tight_layout(pad=1)

        cf1 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf1.pack(fill="x", padx=20, pady=(0, 16))
        embed_figure(cf1, fig1)
        plt.close(fig1)

        # ── Co-occurrence network ─────────────────────────────────
        section_title(sf, "  Co-occurrence Network (strong links)").pack(
            fill="x", padx=20, pady=(4, 8))

        sub_G = nx.Graph()
        threshold = 5000
        for u, v, d in DS.graph.edges(data=True):
            if d["weight"] > threshold:
                sub_G.add_edge(u, v, weight=d["weight"])

        if len(sub_G.nodes) < 2:
            threshold = sorted([d["weight"] for _, _, d in
                                DS.graph.edges(data=True)], reverse=True)[
                                    min(30, DS.graph.number_of_edges()) - 1]
            sub_G = nx.Graph()
            for u, v, d in DS.graph.edges(data=True):
                if d["weight"] >= threshold:
                    sub_G.add_edge(u, v, weight=d["weight"])

        fig2, ax2 = plt.subplots(figsize=(13, 7))
        pos = nx.spring_layout(sub_G, k=1.8, seed=42)
        node_sizes = [DS.pagerank.get(n, 0.001) * 60000
                      for n in sub_G.nodes()]
        node_cols  = [C["accent"] if DS.pagerank.get(n, 0) >= 0.01
                      else C["accent2"] for n in sub_G.nodes()]
        edge_ws    = [d["weight"] / 8000
                      for _, _, d in sub_G.edges(data=True)]

        nx.draw_networkx_nodes(sub_G, pos, node_size=node_sizes,
                               node_color=node_cols, alpha=0.9, ax=ax2)
        nx.draw_networkx_labels(sub_G, pos, font_size=8,
                                font_color=C["bg"], font_weight="bold", ax=ax2)
        nx.draw_networkx_edges(sub_G, pos, width=edge_ws,
                               edge_color=C["border"], alpha=0.5, ax=ax2)
        ax2.set_title(f"Co-occurrence Network (top edges, weight > {threshold:,})",
                      color=C["text"], fontsize=12, pad=10)
        ax2.axis("off")

        legend_items = [
            mpatches.Patch(color=C["accent"],  label="High PageRank"),
            mpatches.Patch(color=C["accent2"], label="Lower PageRank"),
        ]
        ax2.legend(handles=legend_items, loc="lower right",
                   facecolor=C["card"], edgecolor=C["border"],
                   labelcolor=C["muted"])
        fig2.tight_layout()

        cf2 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf2.pack(fill="x", padx=20, pady=(0, 16))
        embed_figure(cf2, fig2)
        plt.close(fig2)

        # ── Recommendations ───────────────────────────────────────
        section_title(sf, "  Strategic Combo Recommendations").pack(
            fill="x", padx=20, pady=(4, 8))

        rec_f = tk.Frame(sf, bg=C["bg"])
        rec_f.pack(fill="x", padx=20, pady=(0, 20))

        top5 = pr_df.head(5)["Meal"].tolist()
        for i, meal in enumerate(top5):
            if meal not in DS.graph:
                continue
            neighbors = DS.graph[meal]
            if not neighbors:
                continue
            best = max(neighbors, key=lambda x: neighbors[x]["weight"])
            weight = neighbors[best]["weight"]

            rec_f.columnconfigure(i, weight=1)
            card = tk.Frame(rec_f, bg=C["card"],
                            highlightbackground=C["accent"], highlightthickness=1,
                            padx=14, pady=14)
            card.grid(row=0, column=i, sticky="nsew", padx=5)
            tk.Label(card, text="🌟", font=("Segoe UI", 18),
                     bg=C["card"]).pack()
            tk.Label(card, text=meal, font=FONT["subhead"],
                     fg=C["accent"], bg=C["card"]).pack(pady=(4, 0))
            tk.Label(card, text=f"+ {best}", font=FONT["body"],
                     fg=C["accent2"], bg=C["card"]).pack()
            tk.Label(card, text=f"Co-ordered {weight:,}×",
                     font=FONT["small"], fg=C["muted"],
                     bg=C["card"]).pack(pady=(4, 0))

        # ── Full table ────────────────────────────────────────────
        section_title(sf, "  Full PageRank Table").pack(
            fill="x", padx=20, pady=(8, 8))

        tbl_f = tk.Frame(sf, bg=C["surface"],
                         highlightbackground=C["border"], highlightthickness=1)
        tbl_f.pack(fill="x", padx=20, pady=(0, 20))

        cols = ["Rank", "Meal", "PageRank Score", "Co-occurrences"]
        tv = ttk.Treeview(tbl_f, columns=cols, show="headings",
                          style="Dark.Treeview", height=14)
        sb = ttk.Scrollbar(tbl_f, orient="vertical", command=tv.yview)
        tv.configure(yscroll=sb.set)
        for c, w in zip(cols, [60, 200, 130, 130]):
            tv.heading(c, text=c)
            tv.column(c, width=w, anchor="center" if c != "Meal" else "w")

        for rank, (_, row_data) in enumerate(pr_df.iterrows(), 1):
            meal = row_data["Meal"]
            cooc = sum(DS.graph[meal][n]["weight"]
                       for n in DS.graph.neighbors(meal)) if meal in DS.graph else 0
            tv.insert("", "end", values=[
                rank, meal,
                f"{row_data['PageRank_Score']:.6f}",
                f"{cooc:,}"
            ])

        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")


# ══════════════════════════════════════════════════════════════════
#  PAGE: BERT SENTIMENT
# ══════════════════════════════════════════════════════════════════
class BertPage(BasePage):
    def _build(self):
        if DS.reviews_df is None:
            tk.Label(self, text="⚠️  Reviews CSV not found.\n"
                                "Expected: data/data/ULTRA_FINAL_REVIEWS.csv",
                     font=FONT["heading"], fg=C["muted"], bg=C["bg"]).pack(
                expand=True)
            return

        df = DS.reviews_df.copy()
        df["Sentiment_Label"] = df["Sentiment"].map({1: "Positive", 0: "Negative"})

        sf = scrolled_frame(self)

        # stats
        total = len(df)
        pos   = (df["Sentiment"] == 1).sum()
        neg   = (df["Sentiment"] == 0).sum()
        avg_r = df["Rating"].mean()

        row = tk.Frame(sf, bg=C["bg"])
        row.pack(fill="x", padx=20, pady=(20, 0))
        for i, (t, v, s, c) in enumerate([
            ("Total Reviews",   f"{total:,}", "across all restaurants",     C["accent"]),
            ("Positive",        f"{pos:,}",   f"{pos/total*100:.1f}%",      C["accent2"]),
            ("Negative",        f"{neg:,}",   f"{neg/total*100:.1f}%",      C["danger"]),
            ("Avg Rating",      f"{avg_r:.2f}", "out of 5 stars",           C["yellow"]),
        ]):
            row.columnconfigure(i, weight=1)
            stat_card(row, t, v, s, c).grid(row=0, column=i, sticky="nsew", padx=6)

        # ── Sentiment by Restaurant ───────────────────────────────
        section_title(sf, "  Sentiment Distribution by Restaurant").pack(
            fill="x", padx=20, pady=(20, 8))

        rest_sent = df.groupby(["Restaurant", "Sentiment_Label"]).size().unstack(fill_value=0)
        fig1, ax1 = plt.subplots(figsize=(13, 4.5))
        x = np.arange(len(rest_sent))
        w = 0.38
        ax1.bar(x - w/2, rest_sent.get("Positive", 0), width=w,
                color=C["accent2"], label="Positive", alpha=0.9)
        ax1.bar(x + w/2, rest_sent.get("Negative", 0), width=w,
                color=C["danger"],  label="Negative", alpha=0.9)
        ax1.set_xticks(x)
        ax1.set_xticklabels(rest_sent.index, rotation=30, ha="right", fontsize=9)
        ax1.set_ylabel("Reviews", color=C["muted"])
        ax1.set_title("Positive vs Negative Reviews per Restaurant",
                      color=C["text"], fontsize=12, pad=10)
        ax1.legend(facecolor=C["card"], edgecolor=C["border"], labelcolor=C["muted"])
        fig1.tight_layout(pad=1)

        cf1 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf1.pack(fill="x", padx=20, pady=(0, 16))
        embed_figure(cf1, fig1)
        plt.close(fig1)

        # ── Rating distribution ───────────────────────────────────
        row2 = tk.Frame(sf, bg=C["bg"])
        row2.pack(fill="x", padx=20, pady=(0, 16))
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        # rating hist
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        rating_c = df["Rating"].value_counts().sort_index()
        ax2.bar(rating_c.index, rating_c.values, color=PALETTES["accent"],
                width=0.65, alpha=0.9)
        ax2.set_xlabel("Rating", color=C["muted"])
        ax2.set_ylabel("Count", color=C["muted"])
        ax2.set_title("Rating Distribution", color=C["text"], fontsize=12, pad=10)
        ax2.set_xticks([1, 2, 3, 4, 5])
        fig2.tight_layout(pad=1)

        cf2 = tk.Frame(row2, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf2.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        embed_figure(cf2, fig2)
        plt.close(fig2)

        # sentiment pie
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.pie([pos, neg], labels=["Positive", "Negative"],
                colors=[C["accent2"], C["danger"]],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"linewidth": 2, "edgecolor": C["surface"]})
        ax3.set_title("Overall Sentiment Split", color=C["text"],
                      fontsize=12, pad=10)
        fig3.tight_layout(pad=1)

        cf3 = tk.Frame(row2, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf3.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        embed_figure(cf3, fig3)
        plt.close(fig3)

        # ── Avg rating by restaurant ──────────────────────────────
        section_title(sf, "  Average Rating per Restaurant").pack(
            fill="x", padx=20, pady=(4, 8))

        avg_rest = df.groupby("Restaurant")["Rating"].mean().sort_values(ascending=True)
        fig4, ax4 = plt.subplots(figsize=(13, 3.8))
        bar_c = [C["accent2"] if v >= 3.5 else C["danger"] for v in avg_rest.values]
        bars4 = ax4.barh(avg_rest.index, avg_rest.values, color=bar_c, height=0.65)
        ax4.bar_label(bars4, fmt="%.2f", padding=4, color=C["muted"], fontsize=8)
        ax4.axvline(3.5, color=C["accent"], linestyle="--", alpha=0.6, label="3.5 threshold")
        ax4.set_xlabel("Average Rating", color=C["muted"])
        ax4.set_xlim(0, 5.5)
        ax4.set_title("Average Rating by Restaurant",
                      color=C["text"], fontsize=12, pad=10)
        ax4.legend(facecolor=C["card"], edgecolor=C["border"], labelcolor=C["muted"])
        fig4.tight_layout(pad=1)

        cf4 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf4.pack(fill="x", padx=20, pady=(0, 16))
        embed_figure(cf4, fig4)
        plt.close(fig4)

        # ── Sentiment heatmap ─────────────────────────────────────
        section_title(sf, "  Sentiment by Restaurant & Rating Heatmap").pack(
            fill="x", padx=20, pady=(4, 8))

        heat = df.groupby(["Restaurant", "Rating"])["Sentiment"].mean().unstack()
        fig5, ax5 = plt.subplots(figsize=(13, 4.5))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn",
                    ax=ax5, vmin=0, vmax=1,
                    linewidths=0.3, linecolor=C["border"],
                    cbar_kws={"label": "% Positive", "shrink": 0.8})
        ax5.set_title("Positive Sentiment % by Restaurant × Rating",
                      color=C["text"], fontsize=12, pad=10)
        ax5.tick_params(axis="x", rotation=0, labelsize=9)
        ax5.tick_params(axis="y", rotation=0, labelsize=9)
        fig5.tight_layout(pad=1)

        cf5 = tk.Frame(sf, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        cf5.pack(fill="x", padx=20, pady=(0, 20))
        embed_figure(cf5, fig5)
        plt.close(fig5)


# ══════════════════════════════════════════════════════════════════
#  PAGE: REVIEWS BROWSER
# ══════════════════════════════════════════════════════════════════
class ReviewsPage(BasePage):
    def _build(self):
        if DS.reviews_df is None:
            tk.Label(self, text="⚠️  Reviews data not found.",
                     font=FONT["heading"], fg=C["muted"], bg=C["bg"]).pack(expand=True)
            return

        # ── Filter bar ────────────────────────────────────────────
        fbar = tk.Frame(self, bg=C["surface"],
                        highlightbackground=C["border"], highlightthickness=1,
                        pady=10, padx=16)
        fbar.pack(fill="x")

        tk.Label(fbar, text="Restaurant:", font=FONT["small"],
                 fg=C["muted"], bg=C["surface"]).pack(side="left")
        rests = ["All"] + sorted(DS.reviews_df["Restaurant"].unique())
        self._rest_var = tk.StringVar(value="All")
        rest_cb = ttk.Combobox(fbar, textvariable=self._rest_var,
                               values=rests, width=14, state="readonly")
        rest_cb.pack(side="left", padx=8)

        tk.Label(fbar, text="Sentiment:", font=FONT["small"],
                 fg=C["muted"], bg=C["surface"]).pack(side="left", padx=(12, 0))
        self._sent_var = tk.StringVar(value="All")
        sent_cb = ttk.Combobox(fbar, textvariable=self._sent_var,
                               values=["All", "Positive", "Negative"],
                               width=10, state="readonly")
        sent_cb.pack(side="left", padx=8)

        tk.Label(fbar, text="Rating:", font=FONT["small"],
                 fg=C["muted"], bg=C["surface"]).pack(side="left", padx=(12, 0))
        self._rat_var = tk.StringVar(value="All")
        rat_cb = ttk.Combobox(fbar, textvariable=self._rat_var,
                              values=["All", "1", "2", "3", "4", "5"],
                              width=6, state="readonly")
        rat_cb.pack(side="left", padx=8)

        tk.Label(fbar, text="Search:", font=FONT["small"],
                 fg=C["muted"], bg=C["surface"]).pack(side="left", padx=(12, 0))
        self._search_var = tk.StringVar()
        tk.Entry(fbar, textvariable=self._search_var, width=18,
                 bg=C["card"], fg=C["text"], insertbackground=C["text"],
                 relief="flat", font=FONT["body"]).pack(side="left", padx=8)

        tk.Button(fbar, text="  Filter  ", font=FONT["small"],
                  fg=C["white"], bg=C["accent"],
                  activebackground="#e55a28", relief="flat",
                  cursor="hand2", command=self._apply_filter).pack(side="left", padx=8)

        self._count_lbl = tk.Label(fbar, text="", font=FONT["small"],
                                   fg=C["muted"], bg=C["surface"])
        self._count_lbl.pack(side="left", padx=8)

        # ── Table ─────────────────────────────────────────────────
        tbl_f = tk.Frame(self, bg=C["surface"])
        tbl_f.pack(fill="both", expand=True, padx=0, pady=0)

        cols = ["#", "Restaurant", "Rating", "Sentiment", "Review"]
        self._tv = ttk.Treeview(tbl_f, columns=cols, show="headings",
                                style="Dark.Treeview")
        sb = ttk.Scrollbar(tbl_f, orient="vertical", command=self._tv.yview)
        self._tv.configure(yscroll=sb.set)

        for c, w in zip(cols, [50, 140, 70, 90, 900]):
            self._tv.heading(c, text=c)
            self._tv.column(c, width=w, anchor="w" if c == "Review" else "center")

        self._tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._tv.tag_configure("pos", foreground=C["accent2"])
        self._tv.tag_configure("neg", foreground=C["danger"])

        self._apply_filter()

    def _apply_filter(self):
        df = DS.reviews_df.copy()
        df["Sentiment_Label"] = df["Sentiment"].map({1: "Positive", 0: "Negative"})

        if self._rest_var.get() != "All":
            df = df[df["Restaurant"] == self._rest_var.get()]
        if self._sent_var.get() != "All":
            df = df[df["Sentiment_Label"] == self._sent_var.get()]
        if self._rat_var.get() != "All":
            df = df[df["Rating"] == int(self._rat_var.get())]

        q = self._search_var.get().strip().lower()
        if q:
            df = df[df["Review"].str.lower().str.contains(q, na=False)]

        self._tv.delete(*self._tv.get_children())
        for i, (_, row) in enumerate(df.head(500).iterrows(), 1):
            tag = "pos" if row["Sentiment"] == 1 else "neg"
            stars = "★" * int(row["Rating"]) + "☆" * (5 - int(row["Rating"]))
            self._tv.insert("", "end", values=[
                i, row["Restaurant"], stars,
                row["Sentiment_Label"], row["Review"][:160]
            ], tags=(tag,))

        self._count_lbl.config(text=f"Showing {min(500, len(df))} / {len(df)} reviews")


# ══════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS / ABOUT
# ══════════════════════════════════════════════════════════════════
class SettingsPage(BasePage):
    def _build(self):
        sf = scrolled_frame(self)

        section_title(sf, "  Project Info").pack(fill="x", padx=20, pady=(20, 12))

        info = [
            ("Project",     "Food Delivery Pattern Analysis"),
            ("Dataset",     "20,000 synthetic orders · 10 restaurants"),
            ("Reviews",     "20,000 customer reviews with sentiment labels"),
            ("Methods",     "Apriori · FP-Growth · PageRank · BERT"),
            ("Libraries",   "pandas · numpy · networkx · matplotlib · seaborn · mlxtend"),
            ("Data Source", "Talabat / Uber Eats (Egypt)"),
        ]
        info_f = tk.Frame(sf, bg=C["card"],
                          highlightbackground=C["border"], highlightthickness=1,
                          padx=24, pady=20)
        info_f.pack(fill="x", padx=20)
        for k, v in info:
            row = tk.Frame(info_f, bg=C["card"])
            row.pack(fill="x", pady=5)
            tk.Label(row, text=f"{k}:", font=("Segoe UI", 10, "bold"),
                     fg=C["muted"], bg=C["card"], width=14, anchor="w").pack(side="left")
            tk.Label(row, text=v, font=FONT["body"],
                     fg=C["text"], bg=C["card"], anchor="w").pack(side="left")

        section_title(sf, "  File Locations").pack(fill="x", padx=20, pady=(24, 12))
        loc_f = tk.Frame(sf, bg=C["card"],
                         highlightbackground=C["border"], highlightthickness=1,
                         padx=24, pady=20)
        loc_f.pack(fill="x", padx=20)

        for label_text, path in [
            ("Orders CSV",  str(ORDERS_CSV)),
            ("Reviews CSV", str(REVIEWS_CSV)),
            ("Outputs",     str(OUTPUTS_DIR)),
        ]:
            row = tk.Frame(loc_f, bg=C["card"])
            row.pack(fill="x", pady=5)
            exists = Path(path).exists()
            icon = "✅" if exists else "❌"
            tk.Label(row, text=f"{icon}  {label_text}:",
                     font=("Segoe UI", 10, "bold"),
                     fg=C["accent2"] if exists else C["danger"],
                     bg=C["card"], width=16, anchor="w").pack(side="left")
            tk.Label(row, text=path, font=FONT["mono"],
                     fg=C["muted"], bg=C["card"], anchor="w").pack(side="left")

        section_title(sf, "  Load Custom Files").pack(fill="x", padx=20, pady=(24, 12))
        btn_f = tk.Frame(sf, bg=C["bg"])
        btn_f.pack(fill="x", padx=20, pady=(0, 20))

        def reload_orders():
            p = filedialog.askopenfilename(
                title="Select Orders CSV",
                filetypes=[("CSV files", "*.csv")])
            if p:
                try:
                    DS.load(orders_path=p)
                    messagebox.showinfo("Reloaded", f"Loaded {p}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        tk.Button(btn_f, text="  📂  Load Orders CSV  ",
                  font=FONT["body"], fg=C["white"], bg=C["accent"],
                  activebackground="#e55a28", relief="flat",
                  cursor="hand2", command=reload_orders).pack(side="left", padx=(0, 10))

        section_title(sf, "  Restaurants & Menus").pack(
            fill="x", padx=20, pady=(16, 8))

        rest_info = {
            "McDonalds":        "Big Mac · McChicken · Quarter Pounder · Fries · Nuggets",
            "KFC":              "Zinger · Twister · Chicken Bucket · Fries · Coleslaw",
            "PizzaHut":         "Pepperoni Pizza · Margherita · Chicken Ranch · Wings",
            "Dominos":          "Pepperoni Pizza · BBQ Chicken Pizza · Potato Wedges",
            "Bazooka":          "Crispy Sandwich · Beef Burger · Fries · Mozzarella Sticks",
            "CookDoor":         "Chicken Sandwich · Shawarma · Fries · Onion Rings",
            "Hardees":          "Thickburger · Chicken Fillet · Fries · Nuggets",
            "TacoBell":         "Taco · Burrito · Nachos · Fries",
            "SpaghettiFactory": "Spaghetti Bolognese · Alfredo Pasta · Garlic Bread · Salad",
            "Zooba":            "Koshary · Falafel Sandwich · Fries · Tamarind Juice",
        }
        for rest, items in rest_info.items():
            rrow = tk.Frame(sf, bg=C["card"],
                            highlightbackground=C["border"], highlightthickness=1,
                            padx=20, pady=10)
            rrow.pack(fill="x", padx=20, pady=3)
            tk.Label(rrow, text=rest, font=FONT["subhead"],
                     fg=C["accent"], bg=C["card"], width=20, anchor="w").pack(side="left")
            tk.Label(rrow, text=items, font=FONT["small"],
                     fg=C["muted"], bg=C["card"]).pack(side="left")


# ══════════════════════════════════════════════════════════════════
#  LOADING SCREEN
# ══════════════════════════════════════════════════════════════════
class SplashScreen(tk.Toplevel):
    def __init__(self, root):
        super().__init__(root)
        self.title("")
        self.geometry("440x300")
        self.resizable(False, False)
        self.configure(bg=C["bg"])
        self.overrideredirect(True)
        self._center()

        tk.Label(self, text="🍔", font=("Segoe UI", 52),
                 bg=C["bg"]).pack(pady=(40, 0))
        tk.Label(self, text="Food Delivery Pattern Analysis",
                 font=FONT["heading"], fg=C["accent"], bg=C["bg"]).pack()
        tk.Label(self, text="Loading data & building models…",
                 font=FONT["small"], fg=C["muted"], bg=C["bg"]).pack(pady=8)

        self._bar_var = tk.DoubleVar(value=0)
        style = ttk.Style(self)
        style.configure("Splash.Horizontal.TProgressbar",
                        troughcolor=C["surface"], background=C["accent"],
                        darkcolor=C["accent"], lightcolor=C["accent"])
        self._bar = ttk.Progressbar(self, variable=self._bar_var,
                                    maximum=100, length=300,
                                    style="Splash.Horizontal.TProgressbar")
        self._bar.pack(pady=16)
        self._lbl = tk.Label(self, text="Initialising…",
                             font=FONT["small"], fg=C["muted"], bg=C["bg"])
        self._lbl.pack()

    def _center(self):
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x = (sw - 440) // 2
        y = (sh - 300) // 2
        self.geometry(f"440x300+{x}+{y}")

    def set(self, pct, msg):
        self._bar_var.set(pct)
        self._lbl.config(text=msg)
        self.update()


# ══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════
class App(tk.Tk):
    PAGES = {
        "overview": ("Overview",       OverviewPage),
        "dataset":  ("Dataset",        DatasetPage),
        "apriori":  ("Association Rules", AprioriPage),
        "pagerank": ("PageRank",        PageRankPage),
        "bert":     ("BERT Sentiment",  BertPage),
        "reviews":  ("Reviews",         ReviewsPage),
        "settings": ("Settings",        SettingsPage),
    }

    def __init__(self):
        super().__init__()
        self.title("Food Delivery Pattern Analysis")
        self.geometry("1280x800")
        self.minsize(1024, 680)
        self.configure(bg=C["bg"])
        self._center()
        self._apply_ttk_styles()
        self._load_data_then_build()

    def _center(self):
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x = (sw - 1280) // 2
        y = (sh - 800) // 2
        self.geometry(f"1280x800+{x}+{y}")

    def _apply_ttk_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background=C["card"], foreground=C["text"],
                        fieldbackground=C["card"], borderwidth=0, rowheight=26)
        style.configure("Dark.Treeview.Heading",
                        background=C["surface"], foreground=C["muted"],
                        relief="flat", borderwidth=0)
        style.map("Dark.Treeview",
                  background=[("selected", C["accent"])],
                  foreground=[("selected", C["white"])])
        style.configure("TCombobox",
                        fieldbackground=C["card"], background=C["card"],
                        foreground=C["text"], selectbackground=C["accent"],
                        borderwidth=0)
        style.configure("TScrollbar",
                        background=C["surface"], troughcolor=C["border"],
                        arrowcolor=C["muted"])

    def _load_data_then_build(self):
        splash = SplashScreen(self)
        self.withdraw()

        def work():
            try:
                splash.set(10, "Reading orders CSV…")
                DS.load()
                splash.set(70, "Building co-occurrence graph…")
                time.sleep(0.2)
                splash.set(90, "Computing PageRank…")
                time.sleep(0.2)
                splash.set(100, "Ready!")
                time.sleep(0.3)
                self.after(0, lambda: self._finish(splash))
            except Exception as e:
                self.after(0, lambda: self._load_error(splash, e))

        threading.Thread(target=work, daemon=True).start()

    def _load_error(self, splash, e):
        splash.destroy()
        self.deiconify()
        msg = (f"Could not load data:\n{e}\n\n"
               f"Make sure the CSV files exist at:\n"
               f"  {ORDERS_CSV}\n  {REVIEWS_CSV}")
        messagebox.showerror("Data Load Error", msg)

    def _finish(self, splash):
        splash.destroy()
        self.deiconify()
        self._build_ui()

    def _build_ui(self):
        # ── Layout ────────────────────────────────────────────────
        self._sidebar = Sidebar(self, self._navigate)
        self._sidebar.pack(side="left", fill="y")

        main = tk.Frame(self, bg=C["bg"])
        main.pack(side="left", fill="both", expand=True)

        self._topbar = Topbar(main)
        self._topbar.pack(fill="x")

        # Page container
        container = tk.Frame(main, bg=C["bg"])
        container.pack(fill="both", expand=True)

        self._pages = {}
        for key, (title, PageClass) in self.PAGES.items():
            page = PageClass(container)
            page.place(relwidth=1, relheight=1)
            self._pages[key] = (title, page)

        self._navigate("overview")

    def _navigate(self, key):
        if key not in self._pages:
            return
        title, page = self._pages[key]
        self._topbar.set_title(title)
        self._sidebar.set_active(key)
        page.activate()


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Ensure matplotlib doesn't open separate windows
    plt.ioff()
    app = App()
    app.mainloop()
