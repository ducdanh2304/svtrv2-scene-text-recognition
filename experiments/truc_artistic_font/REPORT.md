# Báo cáo thí nghiệm — Trúc (artistic font), 21/11

## Câu hỏi cần kiểm tra
Có cách nào cải thiện SVTRv2 trên **chữ nghệ thuật** mà không làm giảm chất lượng trên **chữ thường** không?

## Phương pháp
**EXP-004 — FUSE.** Có cải thiện.
- Chọn giữa SVTRv2 R3_lowlr + TTA và WATERec-RS theo confidence, với δ = 0.15.
- Đây là cách kết hợp ở **mức đầu ra**, không phải can thiệp feature bên trong SVTRv2.
- Phần huấn luyện: fine-tune SVTRv2-B0 thành R3_lowlr (lr 1e-5, 8 epoch). WATERec-RS giữ nguyên, không train lại.

**EXP-005 — GAC.** Chưa cải thiện.
- Đây là hướng **feature-level**.
- Thêm loss kéo đặc trưng ở stage cuối encoder của ảnh nghệ thuật về gần đặc trưng của B0 đóng băng khi B0 nhìn cùng chữ đó vẽ bằng font thường.

## Protocol
| Mục | Nội dung |
|---|---|
| Baseline | SVTRv2-B0 checkpoint gốc (`official_svtrv2_base/best.pth`) |
| Test chính | Union14M-Benchmark artistic, 900 ảnh |
| Đối chứng chữ thường | IIIT5K test, 3000 ảnh |
| Dev | WordArt-Bench, 1511 ảnh; dùng để chọn δ và checkpoint |
| Seed | 0 |
| Metric | Word Accuracy và CER, sau khi chuẩn hoá: chuyển về chữ thường, chỉ giữ `[a-z0-9]`. CER = tổng số phép sửa / tổng số ký tự GT. CI95 tính bằng paired bootstrap. |
| Font train/dev/test | **Không tách theo font**: các bộ dữ liệu không có nhãn font. Thay vào đó tách theo **bộ dữ liệu**: train trên WATER, phát triển trên WordArt-Bench, đánh giá trên Union14M artistic. Ảnh trùng với tập test (md5) đã bị loại khỏi dữ liệu train. |

## Kết quả
| Điều kiện | Số mẫu | Baseline WA | Mới WA | Δ pp | Baseline CER | Mới CER |
|---|---:|---:|---:|---:|---:|---:|
| FUSE · Union ART (test) | 900 | 79.22 | **83.33** | **+4.11** [+2.11, +6.11] | 4.90 | **4.12** |
| FUSE · IIIT5K (chữ thường) | 3000 | 98.70 | 98.80 | +0.10 [−0.23, +0.43] | 0.31 | 0.30 |
| FUSE · WordArt-Bench (dev) | 1511 | 85.37 | 90.27 | +4.90 | 4.67 | 3.41 |
| GAC so với R3 · Union ART | 900 | 78.44 | 78.11 | −0.33 [−1.22, +0.44] | 5.28 | 5.26 |
| GAC so với R3 · IIIT5K | 3000 | 98.47 | 98.33 | −0.13 [−0.37, +0.10] | 0.35 | 0.39 |

## Kiểm tra lỗi (FUSE so với B0)
- **Union ART:** sửa đúng **61**, làm sai thêm **24**, ròng **+37**. So với WATERec-RS: sửa đúng 8, làm sai thêm 0.
- **IIIT5K:** sửa đúng 14, làm sai thêm 11, ròng +3. Chữ thường **không giảm**.
- **Ví dụ tiêu biểu:** dự đoán từng ảnh nằm trong `preds/*.tsv` của artifact. Báo cáo này chưa trích ảnh ví dụ.

## Tái chạy
- **Cell chạy:** `EXP-004_fuse-confidence/kaggle_fuse_release_cell.py` (Kaggle, GPU T4 x2). Cell tự tìm checkpoint R3 trong dataset.
- **Đếm sửa đúng / làm sai thêm:** `fix_break_cell.py`.
- **Config:** `EXP-004_fuse-confidence/configs/RUN-004.yaml` và `EXP-005_gac-glyph-anchor/configs/RUN-005.yaml`.
- **Checkpoint/artifact:** https://www.kaggle.com/datasets/trctrnthanh386/fuse-release (checkpoint R3, config, dự đoán từng ảnh, `summary.csv`).
- **Log:**
  - **R3 và GAC:** `EXP-005_gac-glyph-anchor/logs/RUN-005_cell_output.log`. Đây là output của cell (train R3 đối chứng, train GAC, eval mỗi 500 bước, kết quả cuối), chép lại từ notebook, đã rút gọn các dòng metric trung gian. File `train.log` gốc đã mất khi phiên Kaggle khởi động lại.
  - **FUSE:** `summary.csv` và `preds/` trong artifact.
- **Kết luận:**
  - **FUSE: có cải thiện.** +4.11 pp trên artistic, có ý nghĩa thống kê. Chữ thường giữ nguyên.
  - **GAC: chưa cải thiện.** Không có khác biệt có ý nghĩa so với R3.
  - **Lưu ý:** fine-tune trên dữ liệu WATER tăng điểm trên dev (WordArt) nhưng không tăng trên test (Union). Nguyên nhân có thể là lệch miền giữa hai bộ dữ liệu; **chưa kiểm chứng**.
