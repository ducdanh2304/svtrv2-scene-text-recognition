# FUSE: nhận dạng chữ nghệ thuật (artistic font)

**Artifact (checkpoint, config, dự đoán từng ảnh, summary.csv, log):** https://www.kaggle.com/datasets/trctrnthanh386/fuse-release

## 1. Phương pháp

FUSE kết hợp hai model OCR và chọn kết quả theo độ tin cậy. Không cần train thêm.

| Thành phần | Mô tả | Nguồn checkpoint |
|---|---|---|
| **SVTRv2 R3_lowlr + TTA** | SVTRv2-B0 fine-tune với lr = 1e-5, 8 epoch. Dữ liệu: 20k WATER-T + 50k WATER-R + IIIT5K train ×3, đã lọc trùng md5 với các tập test. TTA: thử thêm khung resize lân cận và xoay ±90° với ảnh dọc, giữ kết quả có confidence cao nhất. | `r3_lowlr_best.pth` (trong gói phát hành) |
| **WATERec-RS** | Model gốc, không train lại | HF `Yesianrohn/WATERec-Models/WATERec-RS.pth` (cell tự tải) |

**Luật kết hợp, δ = 0.15 cố định:**

- Hai model đọc ra cùng chuỗi (sau chuẩn hoá) thì lấy chuỗi đó.
- Nếu khác nhau: lấy WATERec khi `score_WATERec + 0.15 ≥ score_SVTRv2`, ngược lại lấy SVTRv2.

δ được chọn trên WordArt-Bench (dev), dò trong khoảng [−0.5, 0.5] với bước 0.05.

**Baseline:** SVTRv2-B0, checkpoint gốc chưa fine-tune.

## 2. Giao thức đánh giá (đã thống nhất)

| Tập | Số ảnh | Vai trò |
|---|---|---|
| Union14M-Benchmark **artistic** (`topdu/OpenOCR-Data/u14m/artistic`) | 900 | **Test chính** |
| **IIIT5K** test | 3000 | Đối chứng chữ thường |
| WordArt-Bench (`Yesianrohn/WATER-Data`) | 1511 | **Dev**: dùng để chọn δ và checkpoint, **không phải kết quả test** |

**Cách tính số liệu:**

- Chuẩn hoá chuỗi: chuyển về chữ thường và chỉ giữ `[a-z0-9]`.
- Acc tính theo khớp đúng cả từ.
- CER = tổng số phép sửa (edit) / tổng số ký tự GT.
- CI95 tính bằng paired bootstrap với 10,000 lần lấy mẫu.

## 3. Kết quả (bản phát hành, chạy lại trên Kaggle ngày 21/11)

| Model | Union ART acc | Union ART CER | WordArt-Bench acc (dev) | IIIT5K acc |
|---|---|---|---|---|
| Baseline SVTRv2-B0 | 79.22 | 4.90 | 85.37 | 98.70 |
| WATERec-RS | 82.44 | 4.40 | 89.94 | 98.87 |
| R3_lowlr | 78.44 | 5.28 | 86.17 | 98.47 |
| R3_lowlr + TTA | 80.00 | 4.92 | 86.43 | 98.13 |
| **FUSE** | **83.33 (+4.11)** | **4.12 (−0.78)** | 90.27 (+4.90) | **98.80 (+0.10)** |

Ghi chú về cách ra các con số:

- Các số FUSE và R3+TTA ở trên được suy ra từ acc/CER của B0 cộng với Δ trong log.
- Các model còn lại lấy trực tiếp từ log.
- Số chính xác đầy đủ nằm trong `summary.csv`.

**So sánh có CI95 (paired bootstrap):**

| So sánh | Union ART (test) | IIIT5K (đối chứng) |
|---|---|---|
| FUSE − B0 | **+4.11 [+2.11, +6.11]**, ΔCER −0.78 [−1.31, −0.25] | +0.10 [−0.23, +0.43] |
| FUSE − WATERec-RS | **+0.89 [+0.33, +1.56]**, ΔCER −0.29 [−0.51, −0.10] | −0.07 [−0.17, 0.00] |
| FUSE − R3+TTA | **+3.33 [+1.33, +5.33]** | +0.67 [+0.30, +1.03] |

**Số mẫu sửa đúng / làm sai thêm** (đếm bằng `fix_break_cell.py`):

| Tập | So với | Sửa đúng | Làm sai thêm | Ròng |
|---|---|---|---|---|
| Union ART (900) | B0 | 61 | 24 | **+37** |
| Union ART (900) | WATERec-RS | 8 | 0 | **+8** |
| IIIT5K (3000) | B0 | 14 | 11 | +3 |
| IIIT5K (3000) | WATERec-RS | 0 | 2 | −2 |
| WordArt-Bench (dev, 1511) | B0 | 92 | 18 | +74 |
| WordArt-Bench (dev, 1511) | WATERec-RS | 8 | 3 | +5 |

Số ròng khớp với Δacc trong bảng trên. Ví dụ trên Union ART: +37/900 = +4.11 điểm.

**Kết luận:**

- Trên artistic (Union ART), FUSE tốt hơn baseline +4.11 điểm và giảm CER 0.78. Cả hai đều có ý nghĩa thống kê.
- FUSE cũng tốt hơn từng thành phần riêng lẻ (WATERec-RS và R3+TTA), đều có ý nghĩa thống kê.
- Trên chữ thường (IIIT5K), FUSE không giảm so với baseline (+0.10, không có ý nghĩa thống kê).

**Lưu ý:**

- Số trên WordArt-Bench là số dev, vì δ được chọn trên tập này. Không dùng nó như kết quả test.
- Checkpoint R3 là bản train lại ngày 21/11. Vì vậy kết quả lệch nhẹ so với bản báo cáo trước (Union 83.11 → 83.33). Số chính thức là số trong bảng trên.

## 4. Hướng dẫn chạy lại (Kaggle, GPU T4 x2, Internet ON)

1. Tạo notebook mới và bật **GPU T4 x2** và **Internet**.
2. **Add Input:**
   - Checkpoint B0 gốc (đường dẫn `B0_CKPT` trong cell). Nếu checkpoint ở chỗ khác thì sửa biến này.
   - Dataset **https://www.kaggle.com/datasets/trctrnthanh386/fuse-release**. Cell tự tìm `fuse_release/r3_lowlr_best.pth` trong `/kaggle/input`, không cần sửa đường dẫn.
     - Nếu không có checkpoint R3, cell sẽ **tự train lại R3** (khoảng 30 phút, cần tải WATER-T và WATER-R khoảng 18GB vào `/tmp`).
   - IIIT5K: `prathmeshzade/iiit5k-words`.
3. Dán `kaggle_fuse_release_cell.py` vào một cell rồi chạy. Thời gian khoảng 15 phút nếu đã có checkpoint R3:
   - tải dữ liệu test;
   - dự đoán bằng B0, WATERec, R3 và R3+TTA;
   - áp FUSE;
   - in báo cáo kèm CI95.
4. Kết quả nằm trong `/kaggle/working/fuse_release/`:

| File | Nội dung |
|---|---|
| `r3_lowlr_best.pth` | Checkpoint SVTRv2 R3 |
| `r3_lowlr_train_config.yml` | Config train R3 (chỉ có nếu R3 được train trong phiên này) |
| `svtrv2_eval_config.yml` | Config kiến trúc và hậu xử lý dùng khi đánh giá |
| `fuse.json` | δ, luật kết hợp, md5 của checkpoint |
| `summary.csv` | Acc và CER của mọi model trên mọi tập |
| `preds/*.tsv` | Dự đoán từng ảnh (`id`, `gt`, `pred`, `score`) |

Muốn có link tải: bấm **Save Version** cho notebook, sau đó ở tab Output chọn **New Dataset** từ `fuse_release`. Kaggle sẽ tạo link dataset cho gói này.

## 5. Chi phí

| Mục | Mức |
|---|---|
| Model khi chạy | 2 model: SVTRv2 (chạy TTA tối đa 3–5 lần trên ảnh) + WATERec-RS |
| Fine-tune thêm | Không |
| Tốc độ đo trên T4, batch = 1 | B0 khoảng 22 giây / 900 ảnh. FUSE ước tính khoảng 4–6 lần chậm hơn B0, do có TTA và WATERec. |
