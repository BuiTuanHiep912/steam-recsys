"""Demo Streamlit. NGƯỜI A — ngày 11-13 (khung), 18-19 (đầy đủ).

Chạy:
    streamlit run demo/app.py

Nạp mô hình đã huấn luyện từ models/, KHÔNG huấn luyện lại trong demo.
"""

from __future__ import annotations

import streamlit as st


@st.cache_resource
def load_artifacts(models_dir: str = "models", data_dir: str = "data/interim"):
    """Nạp ma trận tương đồng ItemKNN, items.parquet và maps.json.

    Dùng cache_resource để không nạp lại mỗi lần người dùng bấm nút.
    """
    raise NotImplementedError


def recommend_for_selection(liked_i_idx, k: int = 10, price_max=None, platform=None):
    """Sinh gợi ý cho người dùng MỚI (không có trong tập huấn luyện).

    TODO
    ----
    * Đường chính: cộng độ tương đồng ItemKNN từ các game vừa chọn.
    * Loại các game đã chọn khỏi kết quả.
    * Lọc hậu kỳ: giá <= price_max, nền tảng phù hợp.
    * Với mỗi gợi ý, trả thêm game trong lựa chọn giống nó nhất (dòng "vì sao").
    """
    raise NotImplementedError


def main() -> None:
    st.set_page_config(page_title="Gợi ý game Steam", layout="wide")
    st.title("Hệ gợi ý game trên dữ liệu Steam")

    # TODO: ô multiselect chọn 3-5 game theo tên
    # TODO: thanh trượt giá, checkbox nền tảng
    # TODO: nút "Gợi ý cho tôi"
    # TODO: hiển thị top-10 dạng bảng hoặc thẻ, kèm dòng "vì sao"

    st.info("TODO: xem demo/README.md để biết phạm vi tối thiểu")


if __name__ == "__main__":
    main()
