"""Phân tích chuyên sâu. NGƯỜI C — ngày 11-14, 18-20 và 21.

Một bảng điểm số không giải thích được hệ thống hoạt động ra sao. Phần này mới là
nội dung chính của báo cáo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # vẽ ra file, không cần cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.evaluate import run_eval
from src.eval.metrics import ndcg_at_k, novelty_at_k

POP_LABELS = ["phổ biến", "giữa", "đuôi dài"]
BREADTH_LABELS = ["hẹp", "vừa", "rộng"]


# ---------------------------------------------------------------- ngày 11-14

def popularity_groups(item_pop: np.ndarray, n_groups: int = 3) -> np.ndarray:
    """Gán mỗi game vào một nhóm sao cho MỖI NHÓM CHIẾM ~1/n_groups SỐ TƯƠNG TÁC.

    Nhóm 0 = ít game nhưng rất phổ biến; nhóm cuối = rất nhiều game ít người chơi.
    """
    item_pop = np.asarray(item_pop, dtype=np.float64)
    order = np.argsort(-item_pop, kind="stable")
    # Tỷ lệ tích lũy TRƯỚC mỗi game: game thuộc nhóm nơi phần tương tác của nó BẮT ĐẦU.
    # (Dùng tích lũy tính cả chính nó thì game lớn nhất có thể bị đẩy khỏi nhóm đầu.)
    start = (np.cumsum(item_pop[order]) - item_pop[order]) / item_pop.sum()
    cuts = np.linspace(0, 1, n_groups + 1)[1:-1]
    group = np.empty(len(item_pop), dtype=int)
    group[order] = np.searchsorted(cuts, start, side="right")
    return group


def popularity_bias(preds: np.ndarray, item_pop: np.ndarray,
                    n_groups: int = 3) -> pd.DataFrame:
    """Gợi ý rơi vào nhóm game phổ biến hay nhóm đuôi dài?

    So tỷ lệ gợi ý rơi vào từng nhóm (share_recs) với tỷ lệ tương tác của nhóm đó
    trong train (share_train). share_recs > share_train: mô hình KHUẾCH ĐẠI nhóm đó.
    """
    item_pop = np.asarray(item_pop, dtype=np.float64)
    group = popularity_groups(item_pop, n_groups)
    n_in = np.bincount(group, minlength=n_groups)
    rec = np.bincount(group[np.asarray(preds).ravel()], minlength=n_groups)
    labels = POP_LABELS if n_groups == 3 else list(range(n_groups))
    return pd.DataFrame({
        "nhóm": labels,
        "số game": n_in,
        "% catalog": n_in / len(group),
        "% tương tác train": np.bincount(group, weights=item_pop, minlength=n_groups) / item_pop.sum(),
        "% gợi ý": rec / rec.sum(),
    })


def sample_recommendations(model, train_df: pd.DataFrame, items: pd.DataFrame,
                           user_ids: np.ndarray, k: int = 10, n_hist: int = 10) -> str:
    """In lịch sử của vài người dùng cạnh top-10 gợi ý, kèm TÊN GAME và THẺ."""
    title = items.set_index("i_idx")["title"]
    tags = items.set_index("i_idx")["tags"]
    recs = model.recommend(np.asarray(user_ids), k=k)
    lines = []
    for u, row in zip(user_ids, recs):
        h = train_df[train_df.u_idx == u].sort_values("hours", ascending=False)
        lines.append(f"=== user {u} — {len(h)} game trong train")
        lines.append("  LỊCH SỬ (giờ chơi nhiều nhất trước):")
        for i, hrs in zip(h.i_idx.head(n_hist), h.hours.head(n_hist)):
            lines.append(f"    {hrs:7.1f}h  {title[i]}  [{tags[i]}]")
        lines.append(f"  GỢI Ý top-{k} của {model.name}:")
        for r, i in enumerate(row, 1):
            lines.append(f"    {r:2d}. {title[i]}  [{tags[i]}]")
    return "\n".join(lines)


# ---------------------------------------------------------------- ngày 18-20

def split_items_by_release(items: pd.DataFrame, cutoff_ts: int
                           ) -> tuple[np.ndarray, np.ndarray]:
    """(game cũ, game mới) theo date_release so với mốc cắt.

    Dùng cho RQ3: game "mới" là game lọc cộng tác gần như không có dữ liệu để dựa vào.

    Mốc cắt phải khớp với tập model đã học:
      * học trên train          -> cutoff_ts = maps["cutoff_train"]
      * học trên train+valid    -> cutoff_ts = maps["cutoff_valid"]   (chấm test)
    """
    new = items["date_release"].to_numpy() >= cutoff_ts
    idx = items["i_idx"].to_numpy()
    return idx[~new], idx[new]


def rq3_table(models: dict, train_df, test_df, items: pd.DataFrame,
              cutoff_ts: int, metric: str = "ndcg@10") -> pd.DataFrame:
    """Hai cột (game cũ, game mới) x mỗi mô hình một dòng.

    Gọi run_eval hai lần cho mỗi mô hình, mỗi lần với một item_subset khác nhau.
    Khi chấm test: train_df là tập gộp train+valid; MỌI mô hình (kể cả content-based
    của A) học lại trên tập này. cutoff_ts = maps["cutoff_valid"].
    """
    old, new = split_items_by_release(items, cutoff_ts)
    rows = {}
    for name, m in models.items():
        rows[name] = {
            "game cũ": run_eval(m, train_df, test_df, item_subset=old)[metric],
            "game mới": run_eval(m, train_df, test_df, item_subset=new)[metric],
        }
    return pd.DataFrame(rows).T


def profile_entropy(train_df: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    """H(u) = - sum_t pi_t * ln(pi_t), pi_t = tỷ lệ thẻ t trong lịch sử train của u."""
    tags = items.set_index("i_idx")["tags"].fillna("").str.split("|")
    t = (train_df[["u_idx", "i_idx"]]
         .assign(tag=train_df["i_idx"].map(tags))
         .explode("tag"))
    t = t[t["tag"].notna() & (t["tag"] != "")]
    cnt = t.groupby(["u_idx", "tag"]).size()
    p = cnt / cnt.groupby(level=0).transform("sum")
    return (-(p * np.log(p)).groupby(level=0).sum()).rename("entropy")


def metrics_by_breadth(preds, truth, user_ids, entropy: pd.Series, item_pop,
                       n_users: int, n_groups: int = 3, k: int = 10) -> pd.DataFrame:
    """nDCG@10 VÀ novelty@10 theo ba nhóm độ rộng sở thích (chia đều số user)."""
    user_ids = np.asarray(user_ids)
    e = entropy.reindex(user_ids).to_numpy()
    has = ~np.isnan(e)                          # user không có trong train -> bỏ
    labels = BREADTH_LABELS if n_groups == 3 else [str(g) for g in range(n_groups)]
    grp = np.full(len(user_ids), "", dtype=object)
    grp[has] = pd.qcut(e[has], n_groups, labels=labels).astype(str)
    out = []
    for g in labels:
        rows = [r for r in truth if grp[r] == g]
        out.append({
            "nhóm": g,
            "số user": len(rows),
            "entropy TB": float(np.mean(e[rows])),
            f"ndcg@{k}": float(np.mean([ndcg_at_k(preds[r], truth[r], k) for r in rows])),
            f"novelty@{k}": novelty_at_k(preds[rows, :k], item_pop, n_users),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------- ngày 21

COLORS = {"binary": "#4C72B0", "log": "#DD8452", "linear": "#55A868", "capped": "#C44E52"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def _save(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out_dir) / name)
    plt.close(fig)


def fig_rq1(runs: pd.DataFrame, out_dir="reports/figures", metric="ndcg@10"):
    """Hình 1: nDCG@10 (tập kiểm định) theo alpha, mỗi hàm confidence một đường."""
    r = runs[(runs["model"] == "WMF") & (runs["split"] == "valid")]
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, g in r.groupby("scheme"):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g[metric], marker="o", label=s, color=COLORS.get(s))
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 40, 100], ["1", "10", "40", "100"])
    ax.set_xlabel("alpha"); ax.set_ylabel(f"{metric} (tập kiểm định)")
    ax.set_title("RQ1 — nDCG@10 theo alpha"); ax.legend(title="confidence")
    _save(fig, out_dir, "fig1_rq1_ndcg_alpha.png")


def fig_long_tail(item_pop, out_dir="reports/figures"):
    """Hình 2: đường tích lũy — x% game đầu chiếm bao nhiêu % tương tác."""
    p = np.sort(np.asarray(item_pop, dtype=float))[::-1]
    x = np.arange(1, len(p) + 1) / len(p) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, np.cumsum(p) / p.sum() * 100, color=COLORS["binary"])
    ax.set_xlabel("% game (phổ biến nhất trước)"); ax.set_ylabel("% tương tác tích lũy")
    ax.set_title("Phân bố long-tail của catalog")
    _save(fig, out_dir, "fig2_long_tail.png")


def fig_rq3(table: pd.DataFrame, out_dir="reports/figures"):
    """Hình 3: cột nhóm — game cũ vs game mới, mỗi mô hình một màu."""
    fig, ax = plt.subplots(figsize=(6, 4))
    table.T.plot.bar(ax=ax, rot=0, color=list(COLORS.values())[: len(table)])
    ax.set_ylabel("nDCG@10 (tập kiểm thử)"); ax.set_title("RQ3 — game cũ và game mới")
    _save(fig, out_dir, "fig3_rq3.png")
    table.round(4).to_csv(Path(out_dir) / "table_rq3.csv")


def fig_breadth(df: pd.DataFrame, out_dir="reports/figures", k=10):
    """Hình 4: nDCG@10 và novelty@10 theo nhóm độ rộng sở thích, hai trục."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df))
    ax1.bar(x - 0.2, df[f"ndcg@{k}"], 0.4, color=COLORS["binary"], label=f"nDCG@{k}")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, df[f"novelty@{k}"], 0.4, color=COLORS["log"], label=f"novelty@{k}")
    ax1.set_xticks(x, df["nhóm"]); ax1.set_xlabel("độ rộng sở thích")
    ax1.set_ylabel(f"nDCG@{k}"); ax2.set_ylabel(f"novelty@{k}")
    ax2.spines["right"].set_visible(True)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    _save(fig, out_dir, "fig4_breadth.png")


def make_all_figures(runs: pd.DataFrame, item_pop, rq3: pd.DataFrame,
                     breadth: pd.DataFrame, out_dir: str = "reports/figures") -> None:
    """Toàn bộ biểu đồ cho báo cáo, cùng một bảng màu và cùng cỡ chữ."""
    fig_rq1(runs, out_dir)
    fig_long_tail(item_pop, out_dir)
    fig_rq3(rq3, out_dir)
    fig_breadth(breadth, out_dir)