# EXP-005 — GAC (Glyph-Anchored Consistency) — no_improvement

Hướng **feature-level** can thiệp bên trong SVTRv2.

**Phương pháp.** Với mỗi ảnh train nghệ thuật, vẽ lại đúng nhãn bằng font thường (Liberation Sans), cùng kích thước ảnh. Thêm loss vào loss gốc:

- `α·(1 − cos)` giữa đặc trưng của model đang train và đặc trưng của **B0 đóng băng** (B0 nhìn ảnh vẽ lại).
- Áp ở **stage cuối của encoder**, gộp theo chiều cao, chia 8 cột.
- α = 0.3, không dò.

**Đối chứng.** Train giống hệt R3_lowlr: cùng dữ liệu, lr 1e-5, 8 epoch, seed 0, chọn checkpoint trên WordArt-Bench. Chỉ khác loss GAC.

| Tập | R3_lowlr | GAC@R3 | Δ (CI95) |
|---|---|---|---|
| Union ART (900, test) | 78.44 / CER 5.28 | 78.11 / 5.26 | −0.33 [−1.22, +0.44] |
| IIIT5K (3000) | 98.47 | 98.33 | −0.13 [−0.37, +0.10] |
| WordArt-Bench (dev) | 86.17 | 86.43 | +0.26 [−0.27, +0.79] |

**Kết luận.** Không có khác biệt có ý nghĩa so với R3. Thời gian train tăng khoảng 30%.

**Quan sát thêm.** Fine-tune trên dữ liệu WATER tăng điểm WordArt-Bench (dev) nhưng giảm điểm Union ART (test) so với B0, ở cả R3 và GAC. Đây có thể là do lệch miền dữ liệu; chưa kiểm chứng.

Code: `kaggle_gac_full_cell.py`. Config: `configs/RUN-005.yaml`. Log: `logs/RUN-005_cell_output.log`.
