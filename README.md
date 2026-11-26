# Hệ gợi ý game trên dữ liệu Steam

Đồ án môn Hệ gợi ý (IT4613). Nhóm ba người, bốn tuần.

Câu hỏi xuyên suốt: **cách định nghĩa tín hiệu học từ log hành vi ảnh hưởng thế nào
đến chất lượng gợi ý.** Trọng tâm không phải đạt điểm số cao nhất mà là giải thích
được vì sao mô hình gợi ý ra thứ nó gợi ý.

## Câu hỏi nghiên cứu

| Mã | Câu hỏi | Trạng thái | Ai làm |
| --- | --- | --- | --- |
| RQ1 | Giờ chơi làm trọng số tin cậy có tốt hơn tín hiệu nhị phân không, và hàm nào tốt nhất? | Bắt buộc | B chạy, A cấp tín hiệu |
| RQ3 | Với game mới phát hành, content-based có thắng lọc cộng tác không? | Bắt buộc | A dựng mô hình, C đo |
| RQ2 | Chuẩn hóa giờ chơi theo từng game có cải thiện không? | Tùy chọn | B |

## Bốn mô hình

MostPopular (ngưỡng sàn) · ItemKNN (láng giềng) · WMF/ALS (nhân tố ẩn, mô hình chính
của RQ1) · Content-based (dựa trên nội dung, mô hình của RQ3).

Đã cắt khỏi phạm vi: BPR, LightGCN, lai ghép, ablation suy giảm thời gian. Chúng nằm
ở mục *hướng phát triển* của báo cáo.

## Phân công

| Người | Sở hữu | Chương báo cáo |
| --- | --- | --- |
| **A** | `src/data/`, `src/content/`, `demo/` | Dữ liệu, tiền xử lý, tín hiệu, biểu diễn nội dung, demo |
| **B** | `src/models/`, `src/experiments/` | Bốn mô hình và thiết lập thí nghiệm |
| **C** | `src/eval/`, `reports/` | Giới thiệu, đánh giá, kết quả, kết luận |

Mỗi thư mục có `README.md` riêng mô tả việc cần làm, hạn và tiêu chí *Xong khi*.
Không ai commit vào thư mục của người khác.

## Cấu trúc thư mục

```
steam-recsys/
├── README.md               <- file này
├── contracts.py            <- HỢP ĐỒNG GIAO DIỆN, đóng băng sau buổi họp ngày 1
├── make_fake.py            <- sinh dữ liệu giả để ba người chạy song song
├── requirements.txt
├── config/
│   └── default.yaml        <- seed, siêu tham số, đường dẫn
├── data/                   <- không commit, xem .gitignore
│   ├── raw/                <- tải từ Kaggle
│   ├── interim/            <- items.parquet, maps.json
│   └── splits/             <- train/valid/test.parquet
├── models/                 <- mô hình đã huấn luyện, không commit
├── results/                <- runs.csv, không commit
├── reports/
│   └── figures/            <- biểu đồ cho báo cáo
├── src/
│   ├── data/        (A)    <- check_raw.py, preprocess.py, split.py, signals.py
│   ├── content/     (A)    <- content_based.py
│   ├── models/      (B)    <- base.py, popular.py, knn.py, wmf.py
│   ├── experiments/ (B)    <- rq1.py, rq2.py
│   └── eval/        (C)    <- metrics.py, evaluate.py, analyze.py
└── demo/            (A)    <- app.py
```

## Bắt đầu

yêu cầu python 3.14

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python make_fake.py --out data_fake                 # vài giây
```

Từ đây B và C làm việc được ngay trên `data_fake/`, không phải đợi A xử lý
dữ liệu thật.

**Dữ liệu thật** (A làm ở ngày 1): tải từ
[Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam),
giải nén bốn file vào `data/raw/`, rồi kiểm tra:

```bash
python -m src.data.check_raw --raw data/raw
```

Hướng dẫn đầy đủ, kể cả cách dùng Kaggle CLI, nằm trong `src/data/README.md`.
Thư mục `data/` không commit — mỗi người tự tải hoặc chia sẻ qua Drive.

**Ngưỡng kiểm thử trên dữ liệu giả.** Với seed 42, một ItemKNN cosine viết đúng cho
nDCG@10 khoảng **0,25**, còn MostPopular chỉ khoảng **0,04**. Mô hình của bạn chạy
trên bộ này mà không bỏ xa MostPopular thì lỗi nằm ở code, không phải ở dữ liệu.

Dữ liệu giả cũng chứa sẵn 39 game phát hành sau mốc cắt huấn luyện, nên phần RQ3
phát triển được ngay từ tuần 1.

## Lịch và ba điểm đồng bộ

| Tuần | Nội dung | Mốc |
| --- | --- | --- |
| 1 | Dựng đường ống | **Ngày 6–7: pipeline chạy xuyên suốt trên dữ liệu thật** |
| 2 | Thí nghiệm chính RQ1 | **Ngày 14: chốt kết quả RQ1, quyết định có làm RQ2 không** |
| 3 | Content-based, RQ3, demo | **Ngày 21: khóa số liệu, không chạy thêm thí nghiệm** |
| 4 | Viết báo cáo và slide | Ngày 28: chạy thử buổi bảo vệ |

## Quy ước làm việc

- Nhánh riêng cho mỗi người: `feat/data`, `feat/models`, `feat/eval`. Gộp vào `main`
  ở mỗi buổi đồng bộ.
- **Luật nửa ngày:** kẹt quá bốn tiếng thì đăng lên nhóm, không tự vậy.
- Mỗi lần chạy thí nghiệm ghi một dòng vào `results/runs.csv` kèm siêu tham số và seed.
- Seed của cả nhóm là **42**. Không đổi.
- Đổi `contracts.py` phải báo cả ba người.

## Bốn điều phải đúng trước khi tin vào kết quả

1. Đã loại game trong lịch sử huấn luyện khỏi danh sách gợi ý.
2. Xếp hạng trên toàn bộ catalog, không lấy mẫu 100 negative khi đánh giá.
3. Mọi mô hình dùng chung một cách chia train/test và một bộ độ đo.
4. Mô hình thắng cuộc phải vượt MostPopular. Nếu không, có lỗi ở đâu đó.
