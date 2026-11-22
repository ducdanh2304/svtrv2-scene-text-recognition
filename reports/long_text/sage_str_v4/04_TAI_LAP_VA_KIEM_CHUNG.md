# Reproducibility và dữ liệu công khai

## Kaggle / GPU

**Notebook chuẩn:** `experiments/long_text/sage_str_v4/notebooks/SAGE_STR_v4_Kaggle_RunAll.ipynb` (giữ nguyên từ lượt GPU). Kaggle GPU + Internet ON. Notebook có thể tải OpenOCR, dữ liệu và checkpoint khi điều kiện truy cập cho phép; có path overrides ở cell 1 để dùng file attach khi Internet OFF. Bật GPU, Run All; file CSV xuất `/kaggle/working/SAGE_STR_V4_GPU_RESULTS/`.

- `MAX_LTB=1200`, `MAX_EXTERNAL=256` tái tạo cỡ pilot; `0` nghĩa là toàn bộ cohort tương ứng, **không phải một eval độc lập**.
- `OPENOCR_REV=1ccfc6ee...`, config/ckpt SHA256 trong `run_manifest.json` và `provenance.json`.
- `dataset_revision=d59c13643f1d7f92831625fe161e2c19ec3d7f52` trong `dataset_manifest.json`.
- Tham số chia train/calibration/tune/audit/eval và salt lưu trong `v4_training_protocol.json`.
- `v3_control_locked` được huấn luyện lại cùng split; không so trực tiếp với eval split khác của notebook v3 cũ.

## Kiểm tra không GPU

```bash
cd experiments/long_text/sage_str_v4
python -m pip install -r requirements-selector.txt
python scripts/verify_source_parity.py
python scripts/verify_pilot_metrics.py
python -m unittest discover -s tests -v
```

Các script CPU chỉ xác thực **source của notebook, mã chạy lưu trong ZIP GPU, logic bộ chọn và tính toán bảng kết quả**; không thể thay cho việc chạy model thật.

## Artifact lưu kèm GitHub

Lưu mã nguồn, notebook, báo cáo Markdown tiếng Việt, CSV từng eval và trace, manifests, và hai file `joblib` nhỏ từ selector pilot. **Không** lưu raw dataset/image, OpenOCR repo, pretrained backbone checkpoint hoặc cache inference cồng kềnh. `joblib` có tính chất Python pickle, **chỉ mở khi tin tưởng nguồn**; có thể huấn luyện selector lại bằng notebook.

## Science / scientific integrity

Một pilot evaluation split tách ID không biến các benchmark đã dùng để chỉnh method thành test độc lập. SAGE là **frozen backbone + supervised selector**. Không có zero-regression guarantee hay chứng minh tổng quát hóa, và nghiệm thu 12/12 là kiểm tra exploratory nội bộ.
