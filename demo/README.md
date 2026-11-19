# demo — Người A

Một trang Streamlit chạy trên máy. **Không** triển khai lên mạng.

Thời lượng: ngày 11–13 dựng khung, ngày 18–19 bản đầy đủ. Tổng khoảng hai ngày công.
Quá trần thì cắt tính năng, không kéo dài.

## Phạm vi tối thiểu

| Thành phần | Mô tả |
| --- | --- |
| Ô chọn game | Người dùng gõ tên, chọn 3–5 game mình thích |
| Danh sách gợi ý | Top-10 kèm tên, giá, nền tảng, ngày phát hành |
| Bộ lọc | Thanh trượt giá và chọn nền tảng — lọc hậu kỳ kiểu tri thức |
| Dòng "vì sao" | Với mỗi gợi ý, hiện game trong lựa chọn có độ tương đồng cao nhất |

## Điểm kỹ thuật quan trọng

Người xem demo **không có trong tập huấn luyện**, nên không có vector `x_u`. Phải
dùng một trong hai đường:

1. **ItemKNN** — cộng độ tương đồng từ các game vừa chọn. Khuyến nghị dùng làm
   đường chính vì nó cấp luôn dòng *vì sao*.
2. **Fold-in của ALS** — `model.recommend(..., recalculate_user=True)`.

Đây là lý do nên dựng khung demo từ ngày 11 trên ma trận tương đồng ItemKNN mà B
đã xuất từ ngày 5: demo không phụ thuộc việc B có lưu mô hình WMF đúng hạn hay không.

Với **game mới phát hành**, dùng `ContentBased.recommend_for_new_user` của chính bạn.

## Vì sao demo đáng làm hơn là một gánh nặng

Nó buộc nhóm triển khai đường xử lý người dùng mới — và đó là một mục có nội dung
để viết trong báo cáo, không phải công sức bỏ đi.

## Không làm

Đăng nhập, cơ sở dữ liệu, lưu lịch sử, triển khai lên máy chủ, giao diện đẹp.

## Tùy chọn nếu xong sớm

Thanh trượt *khám phá* dùng tái xếp hạng MMR:

    score(i) = lambda * s(u,i) - (1 - lambda) * max_{j đã chọn} sim(i,j)

Khoảng 15 dòng, nhìn rất thuyết phục khi trình bày. Chỉ làm sau ngày 19.
