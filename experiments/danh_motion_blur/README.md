# Danh — nhận dạng chữ dưới motion blur

Phần bàn giao feature-level của Danh gồm hai thí nghiệm và một giao diện suy luận chung:

| Thí nghiệm | Cách chạy | Ảnh gốc | Motion blur nặng | CER blur nặng |
|---|---|---:|---:|---:|
| B0 | Checkpoint SVTRv2-T/RCTC gốc | 98,60% | 80,23% | 8,35% |
| [EXP-001 S2_AUG](EXP-001_s2-motion-aug/) | Train Stage 2 + Stage 3 + RCTC | 98,07% | **89,17%** | 3,57% |
| [EXP-006 A1+A2+A3](EXP-006_stage-adapters/) | Khóa B0, train adapter sau ba Stage | 98,43% | 84,20% | 5,84% |

Các số dùng cùng 3.000 ảnh IIIT5K mỗi điều kiện, chuẩn hóa `RecMetric` của OpenOCR. [Inference API](inference/) nhận **crop chữ** dạng PIL RGB và trả về text/confidence, để Thúy nối vào pipeline end to end. [Sơ đồ](docs/pipeline_adapter_grid.png) và [báo cáo ngắn](docs/report.pdf) giải thích vị trí module, kết quả và thời gian T4.

Checkpoint nguồn đã được audit bằng SHA256:

| Nguồn | SHA256 |
|---|---|
| OpenOCR SVTRv2-T official | `684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6` |
| S2_AUG `latest.pth` | `f7874600e9fe3cd666bccf7012a0b3a85f64f97d00fb1855c8085b7fddff39a8` |
| A1+A2+A3 `final/1+2+3/latest.pth` | `dc59b617ddc9d57454fcb64964b0a70bdc9bb4e844fb0eea9801e3c3f640c66f` |

**Chưa có link tải hai checkpoint cuối trong repo.** Các ZIP audit chứa bảng/log, không chứa trọng số. Theo quy tắc repo, không commit checkpoint `.pth/.pt`; sau khi đăng checkpoint lên Kaggle/Drive, điền URL vào [artifact_manifest.json](artifact_manifest.json), README hai EXP và tracker rồi chuyển trạng thái `blocked` thành `completed`.

Nguồn mô hình: [OpenOCR/SVTRv2](https://github.com/Topdu/OpenOCR), commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`. MotionBlurAug là code từ notebook của nhóm. Adapter A1+A2+A3 là thiết kế thử nghiệm của nhóm, không phải reproduction của paper.
