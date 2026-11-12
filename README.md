# SVTRv2 Improvement Project

Repository chung để nhóm lưu code thí nghiệm và theo dõi tiến độ cải thiện SVTRv2 trên motion blur và artistic font.

## Bảng theo dõi

Mỗi người cập nhật `tracker/experiments.csv` khi bắt đầu, có kết quả hoặc kết thúc một thử nghiệm. Trạng thái dùng thống nhất: `planned`, `running`, `completed`, `blocked`, `no_improvement`.

Quy trình đầy đủ về mã EXP/RUN, branch, kết quả và Pull Request nằm trong [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md).

| Thành viên | Hướng | Thư mục làm việc |
|---|---|---|
| Danh | Tiếp tục thử và cải tiến hướng feature-level đã có tín hiệu tốt trên motion blur | `experiments/danh_motion_blur/` |
| Thúy | Tiếp tục thử và cải tiến hướng đã có tín hiệu tốt trên motion blur | `experiments/thuy_motion_blur/` |
| Thư | Tự tìm, triển khai và đánh giá hướng feature-level trên artistic font | `experiments/thu_artistic_font/` |
| Trúc | Tự tìm, triển khai và đánh giá hướng feature-level trên artistic font | `experiments/truc_artistic_font/` |

## Quy tắc để kết quả so sánh được

- Mỗi kết quả phải ghi baseline/checkpoint, dữ liệu và split, seed, metric, cấu hình và đường dẫn log.
- So sánh với baseline trên cùng ảnh test và cùng cách tiền xử lý. Artistic-font test cần ghi rõ font nào dùng để phát triển và font nào giữ riêng để đánh giá.
- Báo cáo Word Accuracy và CER; kèm số mẫu sửa đúng và số mẫu bị làm sai thêm nếu có.
- Ghi cả kết quả không cải thiện. Không chỉ tải checkpoint mà thiếu cấu hình hoặc log.
- Không commit dataset, ảnh sinh hàng loạt, checkpoint lớn, cache hay token/secret. Lưu chúng ở Kaggle/Drive; commit script, manifest không nhạy cảm, bảng metric và link artifact.

## Quy trình nộp kết quả

1. Tạo branch `member/short-description`.
2. Đặt code, cấu hình và báo cáo trong đúng thư mục thí nghiệm của mình.
3. Cập nhật `tracker/experiments.csv` và `tracker/latest_summary.md`.
4. Mở Pull Request theo mẫu `.github/pull_request_template.md`; điền baseline, thay đổi, metric và link log/checkpoint.
5. Chỉ gộp khi người phụ trách kiểm tra được cách chạy và kết quả.

## Cấu trúc

```text
experiments/       code và báo cáo theo người/hướng
tracker/           bảng tiến độ và kết quả mới nhất
reports/           mẫu báo cáo tuần
baselines/         định nghĩa baseline và số liệu đã xác nhận
```
