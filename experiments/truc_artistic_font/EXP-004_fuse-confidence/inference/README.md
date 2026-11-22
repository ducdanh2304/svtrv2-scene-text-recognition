# Inference FUSE — danh sách file

| File | Ở đâu | Ghi chú |
|---|---|---|
| `r3_lowlr_best.pth` | Kaggle dataset [trctrnthanh386/fuse-release](https://www.kaggle.com/datasets/trctrnthanh386/fuse-release) → `fuse_release/` | Checkpoint SVTRv2 R3_lowlr (84MB). Không commit vào git. |
| `fuse.json` | **Thư mục này** (bản sao từ dataset) | δ = 0.15, luật chọn, md5 của checkpoint R3 |
| `svtrv2_eval_config.yml` | **Thư mục này** (bản sao từ dataset) | **Config model và hậu xử lý của R3** (dùng cho inference; cùng kiến trúc với B0). Khi dùng: sửa `character_dict_path` (đang trỏ tới đường dẫn Kaggle) về `<repo OpenOCR-WATERec>/tools/utils/EN_symbol_dict.txt`. Phần Train/Eval trong file là mặc định của upstream, không dùng khi inference. |
| `r3_lowlr_train_config.yml` | Tạo bằng `python make_r3_train_config.py svtrv2_eval_config.yml r3_lowlr_train_config.yml` | Config train của R3, dựng lại; xem ghi chú bên dưới |
| `waterec_rs_navit_ar.yml` | Thư mục này | Config WATERec-RS (đúng file trong repo upstream, không sửa) |
| `source_lock.json` | Thư mục này | Commit của OpenOCR-WATERec, revision trên HF của WATERec-RS, các gói pip |

**Ghi chú về config train của R3.** File `config.yml` gốc của run R3 không được lưu vào artifact. Script `make_r3_train_config.py` áp lại đúng các tham số mà cell đã dùng:

- lr = 1e-5, 8 epoch, OneCycleLR với warmup 0.5 epoch
- batch 64 mỗi GPU, 2 GPU T4, AMP
- eval mỗi 500 bước trên WordArt-Bench, seed 0

Các tham số này cũng được ghi trong `../configs/RUN-004.yaml`.

**Inference FUSE**

1. Chạy SVTRv2 (`svtrv2_eval_config.yml` + `r3_lowlr_best.pth`) với TTA. Lấy chuỗi đầu CTC và score.
2. Chạy WATERec-RS (`waterec_rs_navit_ar.yml` + `WATERec-RS.pth`). Lấy chuỗi và score.
3. So sánh hai chuỗi sau khi chuẩn hoá (chữ thường, chỉ giữ `[a-z0-9]`):
   - Nếu giống nhau: giữ chuỗi đó.
   - Nếu khác nhau: chọn WATERec nếu `score_W + 0.15 ≥ score_S`, ngược lại chọn SVTRv2.

Code tham chiếu: class `Recognizer` (gồm cả TTA) và hàm `fuse()` trong `../kaggle_fuse_release_cell.py`.
