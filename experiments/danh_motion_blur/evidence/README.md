# Bảng tổng hợp và audit

Các file ở đây ghi kết quả đánh giá **tổng hợp** trên 3.000 ảnh IIIT5K mỗi điều kiện, cấu hình, checkpoint và lịch train. `final_accuracy_cer.csv` ghi word accuracy và CER; `final_paired_vs_B0.csv` và `final_paired_vs_S2_AUG.csv` ghi số ảnh sửa đúng/làm sai thêm. `adapter_grid_audit.json` và `a123_final_train_log.csv` ghi thông tin lần train adapter. `S2_AUG_AUDIT.json` và `S2_AUG_FINAL_COMPARISON.csv` ghi lần train S2; `feature_probes.csv` ghi thống kê feature.

Nguồn: `Danh_adapter_grid_audit_only.zip` do Danh cung cấp (SHA256 `b979d0eeb784c2fd2f6d5f872d9708afbbb39f0564b09a3129a967f8ff9e912e`) và `Danh_S2_Union14M_MotionAug_audit_only.zip` (SHA256 `4ef671c32e5466de4ee6ad24c7bc773eddbb5093a99ce436b71136c274ad4a7a`). Hai ZIP gốc được lưu ngoài GitHub.

**Không dùng bảng paired cũ trong ZIP S2**: thứ tự ảnh giữa hai model khi đó chưa khớp, làm số fixed/broken sai. Số đúng là 299/31 cho S2_AUG và 149/30 cho A1+A2+A3 trên blur nặng, từ lần đánh giá sau đã căn cùng thứ tự ảnh. Repo công bố bảng tổng hợp, không công bố nhãn/dự đoán của từng ảnh test. Các bảng này không thay thế một lần đánh giá độc lập trên dữ liệu mới.
