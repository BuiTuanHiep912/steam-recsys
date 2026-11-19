# src/models — Người B

Bốn mô hình, dùng chung một giao diện. C gọi đúng `fit` và `recommend`, không được
biết gì thêm về bên trong.

## Việc cần làm

### 1. `base.py` — ngày 1–2

Lớp cơ sở xử lý phần chung: nhớ lịch sử để `exclude_seen` hoạt động, kiểm tra kích
thước mảng trả về.

**Xong khi:** một mô hình rỗng trả về mảng đúng kích thước `(len(user_ids), k)`.

### 2. `popular.py` — ngày 3

Đếm số tương tác mỗi game **từ tập huấn luyện**. Không dùng cột `user_reviews`
trong `games.csv` — đó là ảnh chụp năm 2023 và đã bao gồm tương tác trong tập
kiểm thử.

**Xong khi:** chạy trên dữ liệu giả ra top-10 hợp lý.

### 3. `knn.py` — ngày 4–5

`CosineRecommender` của `implicit`, quét `K` trong {20, 50, 100}.

Xuất ma trận tương đồng ra `models/` — A cần nó cho demo từ ngày 11.

**Xong khi:** in ra được vài cặp game tương đồng cao nhất và chúng trông hợp lý
với người đọc.

### 4. `wmf.py` — ngày 8–10

Mô hình chính của RQ1.

```python
AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20, alpha=1.0)
```

**Cạm bẫy:** thư viện `implicit` có tham số `alpha` riêng trong hàm khởi tạo và một
số phiên bản tự nhân ma trận với nó. Hãy tự tính confidence bằng `build_confidence`
của A rồi đặt `alpha=1.0` để tránh nhân hai lần. Kiểm tra lại docstring của đúng
phiên bản đang cài.

**Xong khi:** chạy hết trên dữ liệu thật dưới 10 phút và vượt MostPopular trên nDCG@10.

### 5. Lưu mô hình cho demo — ngày 14

Xuất `item_factors` và `user_factors` của cấu hình thắng ra `models/`.

**Xong khi:** A nạp được và sinh gợi ý mà không cần huấn luyện lại.

### 6. Chương báo cáo — ngày 18–25

Bốn mô hình, công thức, thiết lập thí nghiệm, kết quả RQ1. Bắt đầu viết từ ngày 18
để tuần cuối không dồn.

## Bạn phụ thuộc vào ai

- **A** cho `build_confidence` (RQ1) và `splits/train.parquet`.
- **C** cho `run_eval` để biết mô hình mình chạy tốt hay không.

Chốt chữ ký `build_confidence` với A ngay buổi họp đầu, đừng đợi đến ngày 8.
