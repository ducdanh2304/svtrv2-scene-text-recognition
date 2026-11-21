# Tóm tắt kết quả mới nhất

Cập nhật sau mỗi lần chạy hoàn tất. Chỉ đưa số liệu đã chạy và có log/artifact đối chiếu được; ghi `chưa có` thay vì ước lượng.

| Người | Hướng/phương pháp | Baseline | Kết quả mới | Chênh lệch | Chữ gốc giữ được? | Trạng thái | Link code/log |
|---|---|---:|---:|---:|---|---|---|
| Danh | EXP-001 S2_AUG (Stage 2+3+RCTC) | B0 80,23% / CER 8,35 | **89,17% / CER 3,57** blur nặng | +8,93 pp; sửa 299 / sai thêm 31 | 98,60% → 98,07% | Completed: checkpoint nội dung đã kiểm; SHA ZIP gốc không phục hồi | [code và kết quả](../experiments/danh_motion_blur/EXP-001_s2-motion-aug/) |
| Danh | EXP-006 A1+A2+A3 adapter | B0 80,23% / CER 8,35 | 84,20% / CER 5,84 blur nặng | +3,97 pp; sửa 149 / sai thêm 30 | 98,60% → 98,43% | Completed: checkpoint nội dung đã kiểm; SHA ZIP gốc không phục hồi | [code và kết quả](../experiments/danh_motion_blur/EXP-006_stage-adapters/) |
| Thúy | Motion blur | Chưa nhập | Chưa có | Chưa có | Chưa có | Planned | — |
| Thư | Artistic font | Chưa nhập | Chưa có | Chưa có | Chưa có | Planned | — |
| Trúc | EXP-004 FUSE (R3+TTA vs WATERec-RS, δ=0.15) | Union ART 79.22 / CER 4.90 | 83.33 / CER 4.12 | +4.11 pp [+2.11,+6.11]; sửa 61 / làm sai 24 | IIIT5K 98.70 → 98.80 | Completed | [code](../experiments/truc_artistic_font/EXP-004_fuse-confidence/) · [artifact](https://www.kaggle.com/datasets/trctrnthanh386/fuse-release) |
| Trúc | EXP-005 GAC (feature-level) | R3_lowlr 78.44 | 78.11 | −0.33 pp [−1.22,+0.44] | IIIT5K 98.47 → 98.33 | No improvement | [code](../experiments/truc_artistic_font/EXP-005_gac-glyph-anchor/) |

## Cách đọc

- `delta_pp` là accuracy mới trừ accuracy baseline, tính theo điểm phần trăm.
- Baseline và phương pháp mới phải dùng cùng ảnh test, cùng nhãn và cùng quy tắc chuẩn hóa.
- Với artistic font, ghi kết quả riêng theo font/nhóm font; không chỉ báo cáo trung bình nếu một nhóm giảm mạnh.
