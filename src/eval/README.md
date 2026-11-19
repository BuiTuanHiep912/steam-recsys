# src/eval — Người C

Đánh giá và phân tích. **Cả nhóm phụ thuộc vào `metrics.py`, nên làm nó trước mọi
thứ khác.**

## Việc cần làm

### 1. `metrics.py` — ngày 1–3

nDCG@K, Recall@K, MRR, coverage, novelty.

`evaluate` là **hàm thuần**: không đọc file, không gọi model. Nhờ vậy kiểm thử được
bằng một ví dụ nhỏ tính tay trên giấy.

**Xong khi:** có một ví dụ tính tay và code cho ra đúng con số đó.

### 2. `evaluate.py` — ngày 4–5

`rank_all` và `run_eval`. Nhận một `Recommender` bất kỳ, xếp hạng toàn catalog,
loại game đã tương tác, trả về dict độ đo.

Khai báo sẵn tham số `item_subset` dù đến ngày 18 mới dùng — thêm tham số vào một
hàm đã có ba người gọi phiền hơn nhiều so với khai báo từ đầu.

**Xong khi:** chạy được với cả MostPopular lẫn ItemKNN mà không sửa gì.

### 3. Chạy đánh giá đầy đủ — ngày 8–10

Ba mô hình lọc cộng tác trên dữ liệu thật, cùng một cách chia, cùng một bộ độ đo.

**Xong khi:** có bảng ba dòng để đưa thẳng vào báo cáo.

### 4. Hai phân tích cơ bản — ngày 11–14

- **Thiên lệch phổ biến.** Chia catalog thành ba nhóm theo độ phổ biến, tính tỷ lệ
  gợi ý rơi vào từng nhóm so với tỷ lệ của chúng trong tập huấn luyện.
- **Đọc thử gợi ý.** Chọn năm người dùng, in lịch sử cạnh top-10.

Phân tích thứ hai phát hiện lỗi nhanh hơn mọi độ đo — ví dụ gợi ý toàn bản DLC của
game vừa chơi, hoặc gợi ý giống hệt nhau cho mọi người.

### 5. RQ3 — ngày 18–20

Chia item trong tập kiểm thử thành hai nhóm theo `date_release` so với mốc cắt
huấn luyện. Chấm riêng từng nhóm cho WMF và cho content-based của A.

**Xong khi:** có bảng hai cột (game cũ, game mới) × hai dòng (WMF, content-based).

### 6. Độ rộng sở thích — ngày 18–20

Tính entropy phân bố thẻ trong lịch sử mỗi người dùng, chia ba nhóm, báo nDCG@10
**và** novelty cho từng nhóm.

Dùng lại ma trận TF-IDF của A nên gần như không tốn công thêm.

**Xong khi:** trả lời được câu *mô hình phục vụ người chơi chuyên một dòng tốt hơn
hay người chơi tạp tốt hơn*.

Kết quả dự kiến: nhóm hẹp có nDCG cao nhưng novelty thấp — mô hình đúng một cách
tầm thường, chỉ đoán được thứ người ta đằng nào cũng tìm thấy. Nếu ra như vậy thì
đó là một phát hiện tốt: **nDCG cao không đồng nghĩa với gợi ý có giá trị.**

### 7. Biểu đồ — ngày 21

Đường nDCG@10 theo alpha cho bốn hàm confidence; phân bố long-tail; bảng RQ3; biểu
đồ theo nhóm entropy.

**Xong khi:** cùng một bảng màu, cùng cỡ chữ, xuất ra `reports/figures/`.

### 8. Chương báo cáo và biên tập — ngày 22–28

Viết chương giới thiệu, chương đánh giá, kết luận; sau đó biên tập toàn bộ báo cáo
và ghép slide.

## Vai trò đặc biệt

Bạn quyết định con số nào vào báo cáo và giữ giọng văn thống nhất. Không để ba người
cùng sửa một file ở những ngày cuối.
