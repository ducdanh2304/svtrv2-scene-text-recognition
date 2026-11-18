# Đẩy SAGE-STR v4 lên GitHub (an toàn cho repo cũ)

Archive được đóng gói **theo đường dẫn tương đối từ root của repo**, bổ sung `experiments/long_text/sage_str_v4/` và `reports/long_text/sage_str_v4/` mà **không ghi đè branch Unified v1**.

1. Giải nén file `SAGE_STR_v4_GitHub_Ready.zip` vào **thư mục root** của repository `svtrv2-scene-text-recognition` (sao lưu nếu có file cùng tên từ một lần đóng gói trước). Nếu file nằm ở Downloads và repo trong `~/svtrv2-scene-text-recognition`, chạy:

```bash
cd ~/svtrv2-scene-text-recognition
git status --short
git switch -c feat/cs406-sage-str-v4
unzip ~/Downloads/SAGE_STR_v4_GitHub_Ready.zip -d .
git status --short
git add README_SAGE_STR_v4.md experiments/long_text/sage_str_v4 reports/long_text/sage_str_v4 docs/long_text/SAGE_STR_V4_GITHUB_PUSH.md
git commit -m "Add SAGE-STR v4 frozen-backbone research and audited Kaggle pilot"
git push -u origin feat/cs406-sage-str-v4
```

2. Nếu đã có branch `feat/cs406-sage-str-v4`, thay `git switch -c ...` bằng `git switch feat/cs406-sage-str-v4`. Không `git add .` nếu repo chứa checkpoints/datasets lớn hoặc dữ liệu riêng tư.
3. Kiểm tra file `.ipynb` và các CSV trên GitHub, so source SHA256 trong `run_manifest.json` và chạy lệnh offline validation. Mở Pull Request vào branch chính sau khi duyệt báo cáo.

**Quan trọng:** Không thêm file checkpoint gốc, raw benchmark, API token, data download credentials; các nguồn dữ liệu/giấy phép vẫn thuộc đơn vị phát hành. Không sửa mã trong notebook sau khi đã chốt nếu vẫn gắn nó với bảng kết quả pilot này; nếu sửa, đặt mã phiên bản mới và đánh giá lại.
