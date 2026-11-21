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
2. Lấy checkpoint official từ OpenOCR release. Hai checkpoint S2_AUG và A1+A2+A3 chưa có URL; sau khi nhóm đăng, điền link vào `../artifact_manifest.json`.
3. Chạy `python experiments/danh_motion_blur/inference/scripts/export_checkpoints.py --official-release official_tiny.pth --s2-latest S2_AUG/latest.pth --a123-latest final/1+2+3/latest.pth --openocr OpenOCR --output models`.
4. Script kiểm SHA nguồn, strict-load vào SVTRv2-T/RCTC, chạy smoke và tạo `models/manifest.json`. Chạy `python experiments/danh_motion_blur/inference/tests/smoke_inference.py --openocr OpenOCR --models models` để kiểm lại.

File inference gồm `b0_rctc_state.pt`, `s2_aug_state.pt` và `adapter_a123_state.pt`. **Không commit các file model vào git** theo quy tắc repo; đưa chúng vào kho artifact và ghi URL/SHA ở manifest. `a123` dùng base B0 + ba adapter; `s2_aug` dùng riêng base đã train, không có adapter.

Tiền xử lý trong `model.py` sao chép nhánh evaluation `RatioDataSetTVResize` của OpenOCR: tỉ lệ ảnh quyết định shape, bicubic, chuẩn hóa `[-1,1]`, `padding=False`. Đã kiểm tensor khớp từng phần tử với code gốc ở 5 tỉ lệ ảnh. Đầu ra model là softmax RCTC; text/confidence qua `CTCLabelDecode` gốc. Muốn truyền tensor từ pipeline chung, tensor phải theo đúng quy tắc này rồi gọi `ocr.model(x)`.

Hiện đã kiểm được biến thể B0 và tính đồng nhất của adapter khi khởi tạo. Hai biến thể đã train cần checkpoint thật để kiểm cuối; ZIP audit không đủ chạy forward.
