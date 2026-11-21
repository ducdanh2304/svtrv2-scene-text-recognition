# EXP-006 / RUN-006 — adapter sau Stage 1, 2, 3

**Câu hỏi:** Có thể tăng accuracy ảnh motion blur bằng module sửa feature nhỏ, giữ nguyên toàn bộ SVTRv2-T/RCTC gốc? Đây là thiết kế thử nghiệm của nhóm, không phải reproduction của một paper adapter.

**Vị trí:** `Ảnh → Patch → Stage 1 → A1 → Stage 2 → A2 → Stage 3 → A3 → RCTC → Chữ`. A1/A2/A3 là residual adapter với GroupNorm, depthwise convolution 1×5 theo chiều ngang, bottleneck 1×1, khởi tạo identity. Chỉ 79.648 tham số adapter được train; encoder và RCTC gốc khóa. Khi suy luận không cần nhãn thật hoặc teacher.

**Dữ liệu và lựa chọn:** Cùng Union14M medium 144.677 ảnh và MotionBlurAug như EXP-001. Pilot thử 7 tổ hợp `{A1,A2,A3,A1+A2,A1+A3,A2+A3,A1+A2+A3}` trong 2 epoch. Chọn A1+A2 và A1+A2+A3 theo validation IC13 sạch/blur tổng hợp; train lại từ B0 đủ 12 epoch / 38.304 update rồi mới đo IIIT5K test. A1+A2+A3 là biến thể bàn giao để team thử ghép pipeline.

| Mô hình | Ảnh gốc | Blur nhẹ | Blur vừa | Blur nặng | CER blur nặng |
|---|---:|---:|---:|---:|---:|
| B0 | 98,60% | 98,03% | 94,30% | 80,23% | 8,35% |
| A1+A2 | 98,60% | 98,07% | 94,80% | 83,93% | 5,94% |
| **A1+A2+A3** | 98,43% | 97,90% | 94,50% | **84,20%** | **5,84%** |
| S2_AUG tham chiếu | 98,07% | 97,73% | 95,77% | 89,17% | 3,57% |

Trên blur nặng, A1+A2+A3 hơn B0 **3,97 điểm %**, sửa đúng/làm sai thêm **149/30 ảnh**, nhưng kém S2_AUG **4,97 điểm %**. Gắn thêm A3 chỉ hơn A1+A2 0,27 điểm % blur nặng và giảm ảnh gốc 0,17 điểm %. Vì vậy adapter có tín hiệu hữu ích nhưng không thắng nhánh S2. [results.csv](results.csv) là bảng máy đọc.

**Source/config:** [worker train/eval](code/adapter_grid_trial.py), [config RUN-006](configs/RUN-006.yml), [notebook Kaggle](code/Danh_Stage_Adapter_Grid_2T4.ipynb). Script kiểm shape từng Stage, identity lúc khởi tạo, gradient chỉ ở adapter, loss hữu hạn và thứ tự nhãn trước khi đếm sửa đúng/làm sai. Config chứa đường dẫn Kaggle gốc; notebook là quy trình chuẩn bị dữ liệu/validation đầy đủ.

**Checkpoint:** `final/1+2+3/latest.pth`, SHA256 `dc59b617ddc9d57454fcb64964b0a70bdc9bb4e844fb0eea9801e3c3f640c66f`. **Link tải chưa được cung cấp**, cần điền ở [manifest chung](../artifact_manifest.json). Khi có file, [script export](../inference/scripts/export_checkpoints.py) tạo `adapter_a123_state.pt`; API ở [inference/](../inference/).

Giới hạn: một seed cho model cuối; pilot chọn trên IC13 blur tổng hợp, IIIT5K đã được xem nhiều lần, dải augmentation có tham chiếu protocol test. Khoảng cách feature không thay cho accuracy/CER OCR.
