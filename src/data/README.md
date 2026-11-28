# src/data — Người A

Dữ liệu và tín hiệu. Đây là thư mục cả nhóm phụ thuộc vào, nên trễ ở đây là trễ
của ba người.

## Việc cần làm

### 0. Tải dữ liệu — ngày 1

Bộ dữ liệu: [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
— Anton Kozyriev, 2023. Vài trăm MB khi nén, cần tài khoản Kaggle.

**Cách 1 — tải tay.** Mở trang trên, bấm *Download*, giải nén **bốn file** vào `data/raw/`:

```
data/raw/recommendations.csv     ~41 triệu dòng, file nặng nhất
data/raw/games.csv               ~50 nghìn game
data/raw/users.csv               ~14 triệu người dùng
data/raw/games_metadata.json     mô tả và tags
```

**Cách 2 — dùng Kaggle CLI.** Cần đặt `kaggle.json` vào `~/.kaggle/` trước
(lấy ở Kaggle → Settings → API → Create New Token).

```bash
pip install kaggle
kaggle datasets download -d antonkozyriev/game-recommendations-on-steam -p data/raw
unzip data/raw/*.zip -d data/raw
```

Tải xong chạy ngay:

```bash
python -m src.data.check_raw --raw data/raw
```

Script này chỉ đọc vài dòng đầu nên chạy trong vài giây. Nó bắt ba lỗi hay gặp:
tải thiếu file, giải nén dở dang, và Kaggle đã đổi tên cột so với lúc lên kế hoạch.
Nếu báo cột lạ hoặc thiếu cột, sửa danh sách `EXPECTED` trong `check_raw.py` rồi
**báo cả nhóm** — B và C đang viết code dựa trên tên cột đó.

**Xong khi:** `check_raw.py` in ra *Đủ file, chạy preprocess được.*

**Lưu ý:** `data/` nằm trong `.gitignore`. Không commit dữ liệu, chia sẻ qua Drive
hoặc để mỗi người tự tải.

### 1. `preprocess.py` — ngày 3–5

Đọc `recommendations.csv` (~41 triệu dòng) và sinh ra bộ dữ liệu sạch.

- Đọc có kiểm soát bộ nhớ: chỉ định `dtype` hoặc dùng `polars`. Đọc mặc định bằng
  `pandas` sẽ vượt RAM.
- Lọc `is_recommended == True`.
- Khử trùng cặp (user, game), giữ bản ghi mới nhất theo `date`.
- **Cắt cửa sổ thời gian** theo `config/default.yaml`. Đây là quyết định ảnh hưởng
  tốc độ của cả nhóm — mục tiêu xuống còn vài triệu tương tác.
- Lọc k-core lặp đến khi ổn định (bỏ user rồi lại có game tụt dưới ngưỡng).
- Ánh xạ `user_id` và `app_id` về chỉ số liên tục 0..n−1.
- Ghép `games.csv` và `games_metadata.json` thành `items.parquet`.

Chạy bằng `python -m src.data.preprocess --config config/default.yaml`. Kết quả
gồm `data/interim/interactions.parquet`, `items.parquet` và `maps.json`;
`maps.json` lưu `u_idx_to_user_id`, `i_idx_to_app_id` và `i_idx_to_title` theo
thứ tự chỉ số để tra ngược game. Bảng trước/sau lọc k-core (kèm độ thưa) được
ghi vào `reports/preprocess_stats.csv`.

**Xong khi:** có bảng thống kê trước và sau lọc kèm độ thưa, và `maps.json` tra
ngược được ra tên game.

### 2. `split.py` — ngày 5

Chạy `python -m src.data.split --config config/default.yaml`. Hai mốc cắt được
chọn theo phân bố timestamp để tỷ lệ tương tác gần 80/10/10; các tương tác cùng
timestamp luôn nằm trong cùng một split. Script ghi `train.parquet`,
`valid.parquet`, `test.parquet` vào `data/splits/` và cập nhật hai cutoff vào
`data/interim/maps.json`.

Thông tin `date_release` của từng game được giữ trong
`data/interim/items.parquet` (một dòng mỗi `i_idx`); C dùng bảng này cùng
`cutoff_train`/`cutoff_valid` để chia nhóm game cold-start. Các file interaction
giữ nguyên schema trong `contracts.py`.

**Xong khi:** kiểm tra được không có tương tác nào trong test có mốc thời gian sớm
hơn mốc cắt.

### 3. `signals.py` — ngày 8–10

Bốn hàm confidence theo đúng `build_confidence` trong `contracts.py`.

**Xong khi:** B gọi được và bốn scheme cho ra bốn mảng khác nhau trên cùng dữ liệu.

### 4. Chương báo cáo — ngày 22–25

Bộ dữ liệu, quyết định cắt mẫu, tiền xử lý, thiết kế hai tín hiệu, biểu diễn nội dung.

## Ai phụ thuộc vào bạn

- **B** cần `splits/train.parquet` và `build_confidence` để chạy WMF.
- **C** cần `splits/test.parquet` và `items.parquet` (cột `date_release`) để chấm RQ3.

Đến hết ngày 5 chưa xong thì báo nhóm ngay — B và C vẫn chạy được trên `data_fake/`,
nhưng cần biết để không chờ.

## Cảnh báo

Ba cột `rating`, `positive_ratio`, `user_reviews` trong `games.csv` là ảnh chụp tại
thời điểm tác giả cào dữ liệu năm 2023, **không** phải giá trị tại thời điểm mỗi
review. Dùng chúng làm đặc trưng là rò rỉ dữ liệu tương lai. Độ phổ biến phải đếm
từ tập huấn luyện.
