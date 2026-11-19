# src/content — Người A

Mô hình dựa trên nội dung. Đây là mô hình của **RQ3** và cũng là đường xử lý game
mới trong demo — viết một lần, dùng hai chỗ.

## Việc cần làm

### `content_based.py` — ngày 15–17

Ba bước:

1. **Vector hóa game.** Ghép `tags` và `description` thành một chuỗi, chạy TF-IDF,
   chuẩn hóa về độ dài 1.
2. **Dựng hồ sơ người dùng.** Trung bình có trọng số của vector các game họ đã thích,
   trọng số là `c_ui` từ `signals.py`.
3. **Cho điểm.** Cosine giữa hồ sơ và vector game.

Đóng gói theo đúng giao diện `Recommender` trong `contracts.py`.

**Xong khi:** C chạy `run_eval` lên nó mà không sửa một dòng nào.

## Vì sao mô hình này quan trọng

Đây là mô hình **duy nhất** gợi ý được game vừa phát hành, chưa ai review. Lọc cộng
tác không có cột nào để dựa vào với những game đó.

Đừng đánh giá nó bằng nDCG tổng thể — nó sẽ thua WMF. Hãy so trên **riêng nhóm item
cold-start**, đó mới là chỗ nó có lý do tồn tại. C sẽ chạy phép so này ở ngày 18–20.

## Ai phụ thuộc vào bạn

- **C** cần mô hình này trước ngày 18 để chạy RQ3.
- **Demo** của chính bạn dùng lại nó cho game mới phát hành.

Trễ quá ngày 17 thì báo nhóm: C sẽ chấm RQ3 chỉ với WMF và ItemKNN, kết luận hẹp
lại nhưng vẫn có.
