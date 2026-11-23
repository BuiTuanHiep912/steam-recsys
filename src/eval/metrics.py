"""Bộ độ đo. NGƯỜI C — ngày 1-3. LÀM TRƯỚC MỌI THỨ KHÁC.

evaluate là HÀM THUẦN: không đọc file, không gọi model. Nhờ vậy kiểm thử được bằng
một ví dụ nhỏ tính tay trên giấy — đó là tiêu chí Xong khi của việc này.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def dcg_at_k(hits: np.ndarray, k: int) -> float:
    """DCG = sum_{r=1..k} rel_r / log2(r + 1)

    hits: mảng 0/1 độ dài >= k, phần tử r là 1 nếu vị trí r gợi ý đúng.
    """
    raise NotImplementedError


def ndcg_at_k(pred_row: np.ndarray, truth_set: set[int], k: int) -> float:
    """nDCG = DCG / IDCG.

    IDCG là DCG của trường hợp lý tưởng: min(len(truth_set), k) item đúng xếp đầu.
    """
    raise NotImplementedError


def recall_at_k(pred_row: np.ndarray, truth_set: set[int], k: int) -> float:
    """|top-k ∩ T| / min(|T|, k). Nhất quán với IDCG của nDCG.

    Chia cho min(|T|, k) để người có nhiều game trong test vẫn đạt được 1.
    T rỗng -> 0.
    """
    raise NotImplementedError


def reciprocal_rank(pred_row: np.ndarray, truth_set: set[int]) -> float:
    """1 / thứ hạng của item đúng ĐẦU TIÊN, xét TOÀN BỘ pred_row (k=20 khi gọi
    từ run_eval). Không có item đúng nào thì bằng 0."""
    raise NotImplementedError


def coverage_at_k(preds: np.ndarray, n_items: int) -> float:
    """Tỷ lệ catalog từng xuất hiện trong gợi ý của ít nhất một người dùng."""
    raise NotImplementedError


def novelty_at_k(preds: np.ndarray, item_pop: np.ndarray, n_users: int) -> float:
    """Độ phổ biến nghịch đảo trung bình, có làm trơn +1.

        novelty = trung bình trên mọi ô của preds của  -log2( (pop(i)+1) / (n_users+1) )

    item_pop và n_users đếm TỪ TẬP HUẤN LUYỆN. Làm trơn +1 để game chưa ai chơi
    (game phát hành sau mốc cắt, dùng cho RQ3) có novelty cao nhất nhưng hữu hạn.
    """
    raise NotImplementedError


def evaluate(
    preds: np.ndarray,
    truth: Mapping[int, set[int]],
    k_list: Iterable[int] = (10, 20),
    item_pop: np.ndarray | None = None,
    n_items: int | None = None,
    n_users: int | None = None,
) -> dict[str, float]:
    """Gộp mọi độ đo thành một dict.

    preds : (n_users, k) các i_idx đã xếp hạng, hàng r ứng với người dùng thứ r
    truth : {chỉ số hàng -> tập i_idx đúng trong tập kiểm thử}

    Khóa trả về: ndcg@10, ndcg@20, recall@10, recall@20, mrr,
                 coverage@10, novelty@10

    Giá trị là float thường, không phải np.float32 — để ghi thẳng ra CSV.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Ví dụ tính tay — đây là tiêu chí Xong khi của metrics.py
# ---------------------------------------------------------------------------
# Một người dùng, k = 3, gợi ý [7, 2, 9], item đúng = {2, 5}.
#
#   vị trí 1: item 7  -> sai  -> 0
#   vị trí 2: item 2  -> ĐÚNG -> 1/log2(3) = 0,6309
#   vị trí 3: item 9  -> sai  -> 0
#   DCG  = 0,6309
#   IDCG = 1/log2(2) + 1/log2(3) = 1 + 0,6309 = 1,6309
#   nDCG = 0,6309 / 1,6309 = 0,3868
#   recall@3 = 1/2 = 0,5
#   MRR = 1/2 = 0,5
#
# Viết test khẳng định đúng các con số này trước khi chạy bất cứ mô hình nào.
# Ví dụ 2 — coverage và novelty
# n_items = 5, n_users = 3, item_pop = [3, 1, 0, 2, 1], k = 2
# preds: user 0 -> [0, 1], user 1 -> [0, 3]
#
#   coverage@2 = |{0, 1, 3}| / 5 = 0,6
#   novelty: item 0 -> -log2(4/4) = 0
#            item 1 -> -log2(2/4) = 1
#            item 3 -> -log2(3/4) = 0,4150
#   novelty@2 = (0 + 1 + 0 + 0,4150) / 4 = 0,3538
