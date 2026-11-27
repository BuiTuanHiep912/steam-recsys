"""Test phần ngày 11-14 của analyze.py bằng ví dụ tính tay.
Chạy: python -m src.eval.test_analyze"""
import numpy as np

from src.eval.analyze import popularity_bias, popularity_groups

# item_pop = [50, 30, 10, 5, 5], tổng 100, mỗi nhóm ~1/3 = 33 tương tác.
#   tích lũy TRƯỚC mỗi game: 0, 50, 80, 90, 95 (%)
#   game 0 bắt đầu ở 0%   (< 33%)  -> phổ biến
#   game 1 bắt đầu ở 50%  (33-67%) -> giữa
#   game 2,3,4 bắt đầu >= 67%      -> đuôi dài
POP = np.array([50, 30, 10, 5, 5])


# Mục đích: chia nhóm đúng như tính tay, game lớn nhất phải ở nhóm đầu
def test_groups():
    assert popularity_groups(POP).tolist() == [0, 1, 2, 2, 2]


# Mục đích: tỷ lệ gợi ý đếm đúng
# preds 2 user x 2 game = 4 ô: [0, 1] và [0, 4] -> phổ biến 2/4, giữa 1/4, đuôi 1/4
def test_bias():
    df = popularity_bias(np.array([[0, 1], [0, 4]]), POP)
    assert df["số game"].tolist() == [1, 1, 3]
    assert np.allclose(df["% tương tác train"], [0.5, 0.3, 0.2])
    assert np.allclose(df["% gợi ý"], [0.5, 0.25, 0.25])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK  ", name)