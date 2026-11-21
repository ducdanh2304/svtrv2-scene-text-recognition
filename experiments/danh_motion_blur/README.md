# Danh — nhận dạng chữ dưới motion blur

Phần bàn giao feature-level của Danh gồm hai thí nghiệm và một giao diện suy luận chung:

| Thí nghiệm | Cách chạy | Ảnh gốc | Motion blur nặng | CER blur nặng |
|---|---|---:|---:|---:|
| B0 | Checkpoint SVTRv2-T/RCTC gốc | 98,60% | 80,23% | 8,35% |
| [EXP-001 S2_AUG](EXP-001_s2-motion-aug/) | Train Stage 2 + Stage 3 + RCTC | 98,07% | **89,17%** | 3,57% |
| [EXP-006 A1+A2+A3](EXP-006_stage-adapters/) | Khóa B0, train adapter sau ba Stage | 98,43% | 84,20% | 5,84% |

Các số dùng cùng 3.000 ảnh IIIT5K mỗi điều kiện, chuẩn hóa `RecMetric` của OpenOCR. [Bảng tổng hợp và audit](evidence/) ghi accuracy, CER, số ảnh sửa đúng/làm sai thêm và cấu hình chạy; dữ liệu dự đoán từng ảnh không được công bố trong repo. [Inference API](inference/) nhận **crop chữ** dạng PIL RGB và trả về text/confidence, để Thúy nối vào pipeline end to end. [Sơ đồ](docs/pipeline_adapter_grid.png) và [báo cáo ngắn](docs/report.pdf) giải thích vị trí module, kết quả và thời gian T4.

Checkpoint nguồn đã được audit bằng SHA256:

| Nguồn | SHA256 |
|---|---|
| OpenOCR SVTRv2-T official | `684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6` |
| S2_AUG `latest.pth` | `f7874600e9fe3cd666bccf7012a0b3a85f64f97d00fb1855c8085b7fddff39a8` |
| A1+A2+A3 `final/1+2+3/latest.pth` | `dc59b617ddc9d57454fcb64964b0a70bdc9bb4e844fb0eea9801e3c3f640c66f` |

Hai checkpoint cuối hiện có trong [Kaggle Dataset version 1](https://www.kaggle.com/datasets/cdanht2/checkpoint-svtrv2), nhưng Kaggle đã bung ZIP PyTorch thành các thư mục. [Script xuất model](inference/scripts/export_checkpoints.py) kiểm SHA của toàn bộ bundle và từng entry bên trong, ghép lại checkpoint, strict-load, rồi xuất ba file inference. Vì ZIP gốc đã bị bung, **không thể đối chiếu lại SHA byte của `latest.pth` gốc** từ dataset này; SHA gốc ở bảng trên là từ lần audit trước. Xem [manifest](artifact_manifest.json) và [hướng dẫn](inference/README.md). Repo không commit trọng số `.pth/.pt` theo quy tắc nhóm.

Nguồn mô hình: [OpenOCR/SVTRv2](https://github.com/Topdu/OpenOCR), commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`. MotionBlurAug là code từ notebook của nhóm. Adapter A1+A2+A3 là thiết kế thử nghiệm của nhóm, không phải reproduction của paper.
