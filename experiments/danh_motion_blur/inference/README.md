# API nhận dạng crop chữ cho pipeline

`model.py` cung cấp `load_pipeline(variant, model_dir, openocr_root, device)`. Variant hợp lệ: `b0`, `s2_aug`, `a123`. Đầu vào `predict_crop` là ảnh PIL RGB chỉ chứa **một crop chữ**; pipeline phát hiện/crop chữ của nhóm chạy trước bước này.

```python
from PIL import Image
from experiments.danh_motion_blur.inference.model import load_pipeline

ocr = load_pipeline('s2_aug', model_dir='models', openocr_root='OpenOCR', device='cuda:0')
result = ocr.predict_crop(Image.open('word_crop.png'))
print(result)  # {'text': '...', 'confidence': 0.0..1.0}
```

Hoặc chạy `python experiments/danh_motion_blur/inference/predict.py --variant s2_aug --image word_crop.png --openocr OpenOCR --model-dir models --device cpu`.

## Chuẩn bị

1. Clone OpenOCR và checkout commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`; cài requirements của OpenOCR, `torch`, `torchvision`, `PyYAML`, `Pillow`.
2. Lấy checkpoint official SVTRv2-T từ [OpenOCR model release](https://github.com/Topdu/OpenOCR/blob/1ccfc6ee6161f7133b192e16af3a9265273997ba/configs/rec/svtrv2/readme.md#11-models-and-results); SHA256 phải là `684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6`.
3. Tải [Kaggle Dataset version 1](https://www.kaggle.com/datasets/cdanht2/checkpoint-svtrv2) dạng `archive.zip`, **giữ nguyên bundle này**. Ví dụ: `curl -L 'https://www.kaggle.com/api/v1/datasets/download/cdanht2/checkpoint-svtrv2?datasetVersionNumber=1' -o checkpoint_svtrv2_dataset.zip`.
4. Chạy `python experiments/danh_motion_blur/inference/scripts/export_checkpoints.py --official-release official_tiny.pth --kaggle-bundle checkpoint_svtrv2_dataset.zip --openocr OpenOCR --output models`.
5. Script kiểm SHA của bundle version 1 và byte của **từng entry** trong hai checkpoint đã bị Kaggle bung, ghép lại PyTorch archive, strict-load vào SVTRv2-T/RCTC, chạy smoke và tạo `models/manifest.json`. Chạy `python experiments/danh_motion_blur/inference/tests/smoke_inference.py --openocr OpenOCR --models models` để kiểm lại.

Nếu có hai file `.pth` **nguyên gốc** (chưa bị Kaggle bung), dùng `--s2-latest S2_AUG/latest.pth --a123-latest final/1+2+3/latest.pth` thay `--kaggle-bundle`. Nhánh này kiểm SHA256 của hai file đúng với lần audit ban đầu. Với dataset hiện tại, SHA file `.pth` gốc không thể phục hồi; kiểm định theo toàn bộ nội dung checkpoint và metadata epoch/step/placement. [Kết quả export đã chạy](../docs/kaggle_checkpoint_validation.json) ghi các SHA và giới hạn này.

File inference gồm `b0_rctc_state.pt`, `s2_aug_state.pt` và `adapter_a123_state.pt`. **Không commit các file model vào git** theo quy tắc repo; đưa chúng vào kho artifact và ghi URL/SHA ở manifest. `a123` dùng base B0 + ba adapter; `s2_aug` dùng riêng base đã train, không có adapter.

Tiền xử lý trong `model.py` sao chép nhánh evaluation `RatioDataSetTVResize` của OpenOCR: tỉ lệ ảnh quyết định shape, bicubic, chuẩn hóa `[-1,1]`, `padding=False`. Đã kiểm tensor khớp từng phần tử với code gốc ở 5 tỉ lệ ảnh. Đầu ra model là softmax RCTC; text/confidence qua `CTCLabelDecode` gốc. Muốn truyền tensor từ pipeline chung, tensor phải theo đúng quy tắc này rồi gọi `ocr.model(x)`.

Đã kiểm ba biến thể B0, S2_AUG và A1+A2+A3 chạy forward trên crop chữ mẫu; đây là smoke test, không thay cho bảng accuracy/CER IIIT5K trong README các EXP.
