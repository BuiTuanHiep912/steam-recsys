# reports — Người C biên tập

## Chia chương

| Chương | Người viết |
| --- | --- |
| Giới thiệu, bài toán, câu hỏi nghiên cứu | C |
| Bộ dữ liệu, tiền xử lý, thiết kế tín hiệu, biểu diễn nội dung | A |
| Bốn mô hình và thiết lập thí nghiệm | B |
| Giao thức đánh giá, kết quả RQ1 và RQ3, ba phân tích | C |
| Demo: kiến trúc và đường xử lý người dùng mới | A |
| Kết luận, hạn chế, hướng phát triển | Cả ba cùng viết, C biên tập |

## Hạn chế phải ghi rõ

Bốn điều này nếu không ghi thì hội đồng sẽ hỏi:

1. Tương tác là hành vi **viết review**, không phải mua hay chơi. Kết luận chỉ áp
   dụng cho nhóm người dùng có viết review.
2. Đánh giá hoàn toàn offline. nDCG cao không bảo đảm người dùng thật sẽ bấm vào.
3. Lọc k-core loại bỏ chính nhóm người dùng thưa mà hệ gợi ý cần phục vụ nhất.
4. Dữ liệu không phân biệt được người *chưa khám phá* với người *không muốn khám
   phá* — cả hai để lại dấu vết giống hệt nhau. Phân biệt được cần tín hiệu chỉ có
   khi chạy trực tuyến.

## Bốn câu nên chuẩn bị trước

- Vì sao không dùng RMSE?
- Vì sao chia dữ liệu theo thời gian chứ không ngẫu nhiên?
- Lọc k-core có làm kết quả đẹp lên một cách thiếu trung thực không?
- Mô hình tốt nhất có phải chỉ đang gợi ý game phổ biến không?

Câu cuối dễ bị hỏi nhất, và cũng là lý do bộ độ đo có coverage và novelty.

## Chuẩn bị bảo vệ

Chia việc không có nghĩa là chia hiểu biết. Hội đồng có thể hỏi bất kỳ ai về bất kỳ
phần nào, nên trước buổi bảo vệ cả ba dành một buổi giảng lại phần của mình cho hai
người kia.
