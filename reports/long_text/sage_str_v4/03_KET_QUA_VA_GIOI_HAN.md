# Kết quả GPU và giới hạn claim — SAGE-STR v4

**Nguồn:** ZIP Kaggle `SAGE_STR_V4_GPU_RESULTS.zip` tải lên sau khi chạy GPU; số liệu dưới đây chép từ CSV **trên các image IDs giống nhau trong lượt v4**. Không thay đổi code để chạy lại checkpoint ở môi trường đóng gói.

## 1. Thiết lập kiểm chứng

- **Official frozen SVTRv2-T**, OpenOCR commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- CKPT SHA256 `684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6`; CONFIG SHA256 `f1ab35a0f5e44d3c24c940cdf1a07404068ad9191bad9fe232ad4fcf38bbbe29`.
- LTB 1.200 ảnh và mỗi external dataset tối đa 256 ảnh ở pilot: **2.224 ảnh inference**. Có **461 ảnh eval hợp lệ**; còn lại gồm train/calibration/tune/audit. Mốc hash salt `cs406-sage-v4-locked-splits-v1` không đổi theo cap.
- GT chỉ dùng ở huấn luyện selector và chấm hậu nghiệm, không đưa vào candidate generation/selection khi inference.
- **Toàn bộ nguồn benchmark đã từng được khảo sát trong quá trình phát triển**, nên eval chỉ là *development evaluation*, chưa phải untouched generalization.

## 2. Accuracy và CER trên eval

| Dataset (n) | SVTRv2-T gốc | Unified v1 | SAGE v3 cùng split | SAGE v4 | CER v4 |
|---|---:|---:|---:|---:|---:|
| LTB (241) | 33,20% | 51,45% | 52,70% | **52,70%** | 9,706% |
| Multi-Oriented (58) | 72,41% | 81,03% | 84,48% | **86,21%** | 3,951% |
| Multi-Words (60) | 81,67% | 76,67% | 83,33% | **85,00%** | 4,810% |
| IIIT5K (47) | **97,87%** | 80,85% | **97,87%** | **97,87%** | 0,417% |
| CUTE80 (55) | 92,73% | 60,00% | 94,55% | **96,36%** | 0,738% |

LTB A-Avg: **44,366%** v4, **43,373%** Unified v1, **18,802%** original SVTRv2-T. LTB CER v4 **9,706%** = Unified v1 **9,706%**, còn v3 locked **9,330%** (**tốt hơn v4**). Accuracy v4 và v3 locked đều **127/241**, nhưng các chuỗi dự đoán không nhất thiết trùng nhau.

**LTB bucket:** 26–35: **92/172**; 36–55: **32/55**; 56+: **3/14**. Đặc biệt nhóm 56+ quá nhỏ và accuracy thấp để claim giải quyết văn bản cực dài.

## 3. Kết quả paired (cùng ảnh)

| So sánh (v4 so với Unified v1) | Win | Loss | Net | McNemar exact p |
|---|---:|---:|---:|---:|
| LTB | 4 | 1 | +3 | 0,375 |
| Multi-Oriented | 4 | 1 | +3 | 0,375 |
| Multi-Words | 6 | 1 | +5 | 0,125 |
| IIIT5K | 8 | 0 | +8 | 0,0078125 |
| CUTE80 | 20 | 0 | +20 | 0,0000019 |

Vì p=0,375 trên LTB và Multi-Oriented, **không thể claim vượt Unified ổn định có ý nghĩa thống kê** trong hai tập này. So v4 với SVTRv2-T original trên LTB có **53 win/6 loss**, không có bảo đảm zero harm.

## 4. Audit và reproducibility

Pilot v4 `audit` trên **306 ảnh**: **20 wins, 2 harms**, với `edit_gain=366`. Chính sách pass kiểm tra phát triển trong notebook, không có bảo đảm formal risk/OOD. Split sizes: `train 1003, calibration 230, tune 224, audit 306, eval 461`. Code selector SHA256 `f5ff65b03239c56cd2a7fb5ab39262bde8e10ee86dc25116e9713cf6f94e1c55` đúng với source từ ZIP GPU.

Các file có thể tái tính thống kê: `results/evaluation_predictions.csv`, `evaluation_accuracy_cer_by_dataset.csv`, `evaluation_paired_win_loss.csv`, `ltb_eval_length_buckets.csv` và `ltb_eval_AAvg.csv`. Run `python experiments/long_text/sage_str_v4/scripts/verify_pilot_metrics.py` tại root.

## 5. Điều được claim / không được claim

**Có thể claim có điều kiện:** (i) đóng góp thiết kế selection theo risk/context trên cùng frozen SVTRv2-T, (ii) trong **pilot**, SAGE hạn chế phần lớn regression chữ ngắn của Unified v1 và cải thiện exact-match trên LTB/Multi-Oriented so với original, (iii) Geometry crop giúp long text nhưng ép crop không thích hợp có thể gây hại.

**Không được claim:** phương pháp hoàn thiện tuyệt đối, zero regression, SOTA, verified external generalization, training-free, hiệu quả GPU, conformal guarantee hoặc tất cả component đã được ablation chứng minh độc lập. `source-aware` constraints chỉ dùng cho tuning/evaluation ở phát triển; inference không đọc tên dataset.

## 6. Việc còn lại để chốt nghiên cứu mạnh hơn

1. Đóng băng mã + selector + split, kiểm tra reproducibility nhiều lần/seed trên cùng candidate cache.
2. Chạy **full 4.789 LTB + full external** (báo cáo tách pilot và full).
3. Ablation one-head vs dual; context vs no-context; CER-aware vs exact-only; reference original vs Geometry; từng candidate family trên cùng cache.
4. Đánh giá trên benchmark long-text và short-text **chưa từng dùng để phân tích/chọn phương pháp**; báo cáo latency/VRAM/cost.
