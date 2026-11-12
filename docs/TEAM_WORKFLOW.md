# Quy định làm việc trên GitHub của nhóm SVTRv2

Tài liệu này quy định cách cả nhóm đăng ký, chạy, lưu và báo cáo thí nghiệm để Danh có thể theo dõi tiến độ và kiểm tra kết quả. Áp dụng cho Danh, Thúy, Thư và Trúc.

Repository: [svtrv2-scene-text-recognition](https://github.com/ducdanh2304/svtrv2-scene-text-recognition)

## 1. Nguyên tắc chung

1. Mỗi người làm trong thư mục của mình; không ghi đè code hoặc kết quả của lần chạy trước.
2. Mỗi câu hỏi/phương pháp được theo dõi bằng một `EXP-ID`. Mỗi cấu hình chạy hoặc seed có một `RUN-ID` riêng. Vì vậy một EXP có thể có nhiều RUN.
3. Ghi cả kết quả tốt, không cải thiện, lỗi hoặc bị dừng. Không xóa kết quả để chỉ giữ lại lần chạy đẹp.
4. Mọi so sánh phải ghi rõ baseline, dữ liệu/split, tiền xử lý, seed và checkpoint. Không so hai kết quả chạy trên protocol khác nhau rồi kết luận phương pháp nào tốt hơn.
5. GitHub lưu mã nguồn, cấu hình, bảng kết quả và báo cáo. Dataset, ảnh sinh hàng loạt, checkpoint lớn và log quá lớn để ở Kaggle/Drive; ghi link artifact trong báo cáo.

## 2. Phân công và nơi lưu

| Người | Hướng hiện tại | Thư mục |
|---|---|---|
| Danh | Tiếp tục thử và cải tiến phương pháp đã có tín hiệu tốt trên motion blur | `experiments/danh_motion_blur/` |
| Thúy | Tiếp tục thử và cải tiến phương pháp đã có tín hiệu tốt trên motion blur | `experiments/thuy_motion_blur/` |
| Thư | Tự tìm, triển khai và đánh giá cải thiện feature bên trong SVTRv2 trên artistic font | `experiments/thu_artistic_font/` |
| Trúc | Tự tìm, triển khai và đánh giá cải thiện feature bên trong SVTRv2 trên artistic font | `experiments/truc_artistic_font/` |

## 3. Cách quản lý nhiều thử nghiệm

### 3.1 Cấp mã

- `EXP-ID`: một giả thuyết/phương pháp cần kiểm tra, ví dụ `EXP-005`.
- `RUN-ID`: một cấu hình cụ thể hoặc một seed của EXP, ví dụ `RUN-012`.
- Không tái sử dụng mã đã cấp, kể cả khi thử nghiệm thất bại.
- Tăng số tiếp theo trong `tracker/experiments.csv`; nếu làm đồng thời, các thành viên nhắn nhau để tránh cấp trùng.

Ví dụ: `EXP-005` kiểm tra một phương pháp điều chỉnh feature. `RUN-012` là seed 2026; `RUN-013` là seed 2027 cho cùng phương pháp. Nếu đổi sang một phương pháp/giả thuyết khác, tạo EXP mới.

### 3.2 Tạo thư mục

Mỗi EXP có một thư mục con bên trong thư mục của người phụ trách:

```text
experiments/<member_topic>/EXP-005_short-method-name/
├── README.md       # câu hỏi, phương pháp, cách chạy và kết luận
├── code/           # script hoặc notebook cần thiết
├── configs/        # cấu hình của các RUN
└── results.csv     # một dòng cho mỗi RUN, nếu có nhiều RUN
```

Đặt tên file cấu hình theo RUN-ID, ví dụ `configs/RUN-012.yaml`. Không lưu bản sao dataset hoặc checkpoint nặng trong thư mục này.

## 4. Vòng đời của một thí nghiệm

### Bước 1 — Đăng ký trước khi chạy

Thêm dòng vào `tracker/experiments.csv` với `EXP-ID`, `RUN-ID`, người phụ trách, chủ đề, phương pháp dự kiến, baseline, dữ liệu/split, seed và trạng thái `planned`. Tạo thư mục EXP và ghi câu hỏi cần kiểm tra trong `README.md`.

### Bước 2 — Đánh dấu đang chạy

Khi bắt đầu chạy, đổi trạng thái sang `running`; điền đường dẫn code/config và ngày cập nhật. Mỗi lần đổi cấu hình đáng kể hoặc seed mới phải có RUN-ID mới.

### Bước 3 — Lưu kết quả ngay sau khi chạy

Ghi kết quả thực tế vào CSV và `README.md`, gồm:

- Word Accuracy và CER của baseline, phương pháp mới, cùng mức chênh lệch.
- Số ảnh được sửa đúng và số ảnh bị làm sai thêm, nếu có đối chiếu theo từng ảnh.
- Tên checkpoint, dataset/split, số mẫu, seed, preprocessing và metric.
- Link Kaggle/Drive tới notebook đã chạy, log và checkpoint.
- Trạng thái cuối: `completed`, `no_improvement` hoặc `blocked`.

Không ghi số ước lượng. Nếu một metric chưa đo, để trống và giải thích trong ghi chú.

### Bước 4 — Cập nhật trang tổng hợp

Thêm kết quả mới nhất vào `tracker/latest_summary.md`. Trang này là bảng xem nhanh; `tracker/experiments.csv` và báo cáo trong thư mục EXP là lịch sử chi tiết.

## 5. Chuẩn so sánh

- Dùng cùng checkpoint SVTRv2 gốc, tập ảnh test, nhãn, quy tắc chuẩn hóa, preprocessing và cách giải mã cho baseline và phương pháp mới.
- Motion blur: ghi rõ loại/mức blur, cách sinh ảnh và số ảnh ở mỗi điều kiện.
- Artistic font: ghi rõ nguồn font, font dùng khi phát triển và font giữ riêng để test; báo cáo theo từng nhóm font và cả tổng thể.
- Luôn báo cáo kết quả trên ảnh/chữ gốc làm đối chứng để phát hiện phương pháp làm giảm năng lực nền.
- Ghi accuracy theo phần trăm và mức chênh lệch theo điểm phần trăm (`pp`); CER càng thấp càng tốt.
- Chỉ kết luận khi có đủ số mẫu, cùng protocol và có log/artifact đối chiếu được. Ảnh minh họa không thay thế bảng số liệu.

## 6. Branch, commit và Pull Request

1. Đồng bộ `main` trước khi bắt đầu.
2. Tạo branch cho EXP, ví dụ `danh/exp-005-motion-feature` hoặc `thu/exp-008-artistic-font`.
3. Commit theo nội dung cụ thể, ví dụ `EXP-005: add feature experiment config`.
4. Mở Pull Request vào `main`; dùng mẫu `.github/pull_request_template.md`.
5. Một PR nên tập trung vào một EXP hoặc một nhóm RUN liên quan; không đưa thay đổi của nhiều hướng không liên quan vào cùng PR.
6. Danh kiểm tra baseline, protocol, số liệu, link artifact và khả năng chạy lại trước khi merge. Nếu cần sửa, cập nhật chính PR đó.
7. Không commit trực tiếp vào `main`.

Nếu nhóm chưa quen branch/PR, có thể dùng GitHub Desktop để tạo branch, commit, push và mở PR bằng giao diện.

## 7. Quy tắc bảo vệ dữ liệu và repo

- Không đưa mật khẩu, token, API key, dữ liệu cá nhân hoặc file `.env` lên GitHub.
- Không commit dataset, archive, ảnh sinh hàng loạt, checkpoint lớn hoặc cache. `.gitignore` đã loại các loại file phổ biến; vẫn phải kiểm tra danh sách file trước khi commit.
- Không thay đổi quyền truy cập hoặc mời người ngoài nhóm nếu chưa thống nhất.
- Nếu lỡ commit secret hoặc dữ liệu nhạy cảm, báo Danh ngay; xóa file khỏi commit đơn thuần có thể chưa xóa khỏi lịch sử Git.

## 8. Trạng thái dùng trong tracker

| Trạng thái | Ý nghĩa |
|---|---|
| `planned` | Đã đăng ký, chưa chạy |
| `running` | Đang chạy hoặc đang phân tích kết quả |
| `completed` | Đã chạy và bàn giao đủ code/config/số liệu |
| `no_improvement` | Đã chạy đầy đủ nhưng không cải thiện hoặc làm giảm kết quả |
| `blocked` | Chưa thể hoàn tất; phải ghi lỗi cụ thể và bước đã thử |

## 9. Checklist trước khi nộp PR

- [ ] EXP-ID và RUN-ID không trùng; tracker đã cập nhật.
- [ ] Có README mô tả câu hỏi, phương pháp, baseline và cách chạy.
- [ ] Code/notebook và config đủ để người khác hiểu, chạy lại.
- [ ] Bảng có số mẫu, seed, Word Accuracy, CER và mức chênh lệch.
- [ ] Có link log/checkpoint/artifact; artifact không nằm trong Git nếu quá lớn.
- [ ] Đã báo cáo cả ảnh gốc và các điều kiện mục tiêu.
- [ ] Đã ghi rõ hạn chế hoặc lý do nếu kết quả thất bại/chưa kết luận.
- [ ] Không có dataset, checkpoint lớn, secret hoặc file tạm bị đưa vào PR.
