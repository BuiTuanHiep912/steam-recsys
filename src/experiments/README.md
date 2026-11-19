# src/experiments — Người B

Các kịch bản chạy thí nghiệm. Mỗi lần chạy ghi một dòng vào `results/runs.csv`.

## Việc cần làm

### 1. `rq1.py` — ngày 11–14

Thí nghiệm chính của đồ án.

Bốn hàm confidence × alpha trong {1, 10, 40, 100} = **16 lần chạy**. Mọi siêu tham
số khác cố định: `factors=64`, `regularization=0.05`, `iterations=20`, `seed=42`.

Quy trình bắt buộc:

1. Chọn cấu hình tốt nhất **trên tập kiểm định**.
2. Báo cáo **một lần duy nhất** trên tập kiểm thử.

Dò siêu tham số trên tập kiểm thử sẽ làm con số cuối mất giá trị.

**Xong khi:** `results/runs.csv` có đủ 16 dòng kèm siêu tham số và seed.

### 2. `rq2.py` — ngày 15–17, **tùy chọn**

Chạy lại cấu hình thắng của RQ1 với giờ chơi chuẩn hóa theo trung vị của từng game.

Chỉ làm nếu RQ1 xong trước ngày 15. Quyết định dứt khoát ở buổi đồng bộ ngày 14,
không để lơ lửng.

## Đọc kết quả thế nào

Đừng chỉ ghi giá trị tốt nhất — vẽ đường cong nDCG@10 theo alpha, mỗi hàm confidence
một đường (C dựng biểu đồ, bạn cấp số liệu).

- Đường cong **phẳng** nghĩa là mô hình không nhạy với alpha. Bản thân đó là một
  phát hiện đáng viết.
- Đường cong **nhọn** nghĩa là alpha quan trọng, nên dò mịn hơn quanh đỉnh.
- Đỉnh rơi vào **biên** của dải quét (ví dụ alpha = 100 tốt nhất) thì phải mở rộng
  dải và chạy tiếp — giá trị tối ưu thật có thể nằm ngoài.
