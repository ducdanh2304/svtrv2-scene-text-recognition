# EXP-001 / RUN-001 — S2 với MotionBlurAug

**Câu hỏi:** Khi gặp motion blur, train tiếp feature extractor từ Stage 2 có giúp SVTRv2-T/RCTC đọc chữ tốt hơn không? Đây là ablation của nhóm dựa trên kiến trúc [SVTRv2/OpenOCR](https://github.com/Topdu/OpenOCR), không phải một module mới từ paper.

**Vị trí train:** Patch embedding và Stage 1 giữ trọng số checkpoint official; **Stage 2, Stage 3, đầu RCTC** được train. Không thêm adapter. Từ checkpoint SVTRv2-T official GTC+RCTC, chỉ lấy encoder và nhánh RCTC, strict-load. CTC loss/nhãn và evaluation dùng OpenOCR commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`.

**Dữ liệu và lịch:** Union14M-L-Filter medium 144.677 ảnh; augmentation `PARSeqAugPIL`, sau đó MotionBlurAug của nhóm với xác suất 0,5, độ dài kernel bằng 0,05294–0,31765 chiều cao ảnh, góc −10° đến 160°. Train 12 epoch / 38.304 update / batch 48 trên 2×T4. Đối chứng S2_NOAUG dùng cùng lịch nhưng không có MotionBlurAug.

| Mô hình | Ảnh gốc | Blur nhẹ | Blur vừa | Blur nặng | CER blur nặng |
|---|---:|---:|---:|---:|---:|
| B0 official | 98,60% | 98,03% | 94,30% | 80,23% | 8,35% |
| S2_NOAUG | 97,93% | 97,60% | 93,57% | 78,47% | Chưa đo |
| **S2_AUG** | 98,07% | 97,73% | 95,77% | **89,17%** | **3,57%** |

Tất cả đo trên **cùng 3.000 ảnh IIIT5K test mỗi điều kiện**, word accuracy/CER theo `RecMetric` OpenOCR. S2_AUG hơn B0 **8,93 điểm %** blur nặng; hơn S2_NOAUG **10,70 điểm %**. Số sửa đúng/làm sai thêm so với B0, tính từ lượt đánh giá mới có cùng thứ tự ảnh: **299/31** ở blur nặng. Bảng chi tiết ở [results.csv](results.csv). Dải augmentation được đặt dựa trên manifest blur của nhóm, nên test này phản ánh protocol đã biết; cần tập mới để khẳng định tổng quát.

**Source/config:** [script S2](code/s2_train_pair.py), [MotionBlurAug](code/motion_blur_aug.py), [config RUN-001](configs/RUN-001.yml) và [notebook Kaggle đầy đủ](code/Danh_S2_Union14M_MotionAug_2T4.ipynb). Config ghi lại đường dẫn Kaggle gốc; đổi đường dẫn dữ liệu/checkpoint trên máy khác. Notebook kiểm SHA release, số mẫu, gradient vùng được train, baseline và xuất audit. Khi chạy riêng script S2, đặt OpenOCR tại `EXP-001_s2-motion-aug/OpenOCR` như đường dẫn import trong script; trước khi train, cần [đăng ký augmentation vào OpenOCR](code/install_motion_blur.py).

**Checkpoint để ghép pipeline:** `S2_AUG/latest.pth`, SHA256 `f7874600e9fe3cd666bccf7012a0b3a85f64f97d00fb1855c8085b7fddff39a8`. **Link tải chưa được cung cấp**; cần điền vào [manifest chung](../artifact_manifest.json). Sau khi có file, [script export](../inference/scripts/export_checkpoints.py) tạo `s2_aug_state.pt` không chứa optimizer. API ở [inference/](../inference/).
