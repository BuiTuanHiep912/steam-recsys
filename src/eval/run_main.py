"""Bảng kết quả chính. NGƯỜI C — ngày 6-7 (2 mô hình) và 8-10 (3 mô hình).

Chạy:
    python -m src.eval.run_main --data data_fake --split valid
    python -m src.eval.run_main --data data --split valid --models pop,knn,wmf
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.eval.evaluate import log_run, run_eval

SEED = 42


def load(data_dir: str):
    maps = json.load(open(f"{data_dir}/interim/maps.json"))
    splits = {s: pd.read_parquet(f"{data_dir}/splits/{s}.parquet")
              for s in ("train", "valid", "test")}
    return maps, splits


def fit_and_target(splits: dict, maps: dict, split: str):
    """Quyết định của nhóm: tập học phụ thuộc tập được chấm.

        chấm valid -> học trên train         , train_set="train"      , cutoff_train
        chấm test  -> học lại trên train+valid, train_set="train+valid", cutoff_valid

    Trả về (fit_df, target_df, train_set, cutoff_ts). MỌI script dùng hàm này để
    không ai tự ghép train+valid theo một cách khác.
    """
    if split == "valid":
        return splits["train"], splits["valid"], "train", maps["cutoff_train"]
    if split == "test":
        fit = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
        n_dup = int(fit.duplicated(["u_idx", "i_idx"]).sum())
        if n_dup:
            # csr_matrix sẽ CỘNG DỒN confidence của cặp trùng -> WMF sai âm thầm.
            raise ValueError(f"train+valid có {n_dup} cặp (u_idx, i_idx) trùng — báo A")
        return fit, splits["test"], "train+valid", maps["cutoff_valid"]
    raise ValueError(f"split phải là 'valid' hoặc 'test', nhận {split!r}")


def get_build_confidence():
    """Dùng bản thật của A nếu đã cài, chưa thì dùng stub của B (cùng công thức)."""
    from src.data.signals import build_confidence
    try:
        build_confidence(pd.DataFrame({"hours": np.array([1.0], dtype=np.float32)}))
        return build_confidence
    except NotImplementedError:
        from src.experiments.confidence_stub import build_confidence as stub
        print("! src/data/signals.py chưa xong -> dùng confidence_stub")
        return stub


def make_models(names, n_users, n_items, knn_k, scheme, alpha):
    """Trả về [(model, confidence-hay-None, config)] — cùng MỘT cách gọi cho mọi mô hình."""
    out = []
    for n in names:
        if n == "pop":
            from src.models.popular import MostPopular
            out.append((MostPopular(n_users, n_items), None, {}))
        elif n == "knn":
            from src.models.knn import ItemKNN
            out.append((ItemKNN(n_users, n_items, k_neighbors=knn_k), None, {"k_neighbors": knn_k}))
        elif n == "wmf":
            from src.models.wmf import WMF
            out.append((WMF(n_users, n_items, seed=SEED), "conf",
                        {"scheme": scheme, "alpha": alpha, "factors": 64,
                         "regularization": 0.05, "iterations": 20}))
        else:
            raise ValueError(n)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_fake")
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--models", default="pop,knn")
    p.add_argument("--knn-k", type=int, default=50)
    p.add_argument("--scheme", default="log")
    p.add_argument("--alpha", type=float, default=40.0)
    p.add_argument("--out", default="results/runs.csv")
    a = p.parse_args()

    maps, sp = load(a.data)
    fit_df, target, train_set, _ = fit_and_target(sp, maps, a.split)
    print(f"học trên {train_set} ({len(fit_df):,} dòng)  chấm {a.split} ({len(target):,} dòng)  "
          f"user {a.split} không có trong tập học: {(~target.u_idx.isin(fit_df.u_idx)).mean():.1%}")

    rows = []
    for model, conf, cfg in make_models(a.models.split(","), maps["n_users"], maps["n_items"],
                                        a.knn_k, a.scheme, a.alpha):
        confidence = None
        if conf == "conf":
            confidence = get_build_confidence()(fit_df, scheme=a.scheme, alpha=a.alpha)
        t0 = time.time()
        model.fit(fit_df, confidence=confidence)
        # fit_df cũng là "train" của run_eval: item_pop, n_users, game đã xem đều đếm từ nó
        res = run_eval(model, fit_df, target)
        sec = round(time.time() - t0, 1)
        log_run(res, {"model": model.name, "data": a.data, "split": a.split,
                      "train_set": train_set, "seed": SEED, "seconds": sec, **cfg}, a.out)
        rows.append({"model": model.name, **res, "seconds": sec})

    table = pd.DataFrame(rows).round(4)
    print(table.to_string(index=False))
    table.to_csv(f"results/table_main_{a.split}.csv", index=False)


if __name__ == "__main__":
    main()
