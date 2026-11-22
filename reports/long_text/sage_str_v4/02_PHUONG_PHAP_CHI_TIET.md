# SAGE-STR v4 — Phương pháp đề xuất (đối chiếu code thực nghiệm)

**Tên:** Contextual Risk-Aware Adaptive Inference for Scene Text Recognition.
**Phạm vi:** phương pháp *meta-inference* trên recognizer **SVTRv2-T frozen**, có **learned selection head**. Đây là phát triển dựa trên bằng chứng thực nghiệm, không phải kiến trúc backbone OCR hoặc decoder mới.

## 1. Vấn đề và giả thuyết

Nhận dạng CTC một lần trên ảnh có giới hạn độ phân giải/trục thời gian có thể bỏ sót văn bản dài; ngược lại, chia crop hoặc xoay cưỡng bức có thể phá hình dạng/ngữ cảnh của từ ngắn. Các thử nghiệm full LTB và external (xem `01_HANH_TRINH_THU_NGHIEM.md`) chứng minh **lợi ích và tác hại cùng tồn tại tùy ảnh**. Vấn đề nghiên cứu là **chọn giả thuyết thay thế nào để cải thiện dự đoán tham chiếu, đồng thời kiểm soát rủi ro làm sai dự đoán vốn đúng**.

## 2. Biểu diễn thống nhất

Cho ảnh `I`, recognizer `f_theta` với tham số đóng băng tạo dự đoán nguyên ảnh `h0`. Các phép biến đổi cho phép tạo tập hữu hạn các giả thuyết `H(I)` sử dụng **cùng backbone**. Mỗi giả thuyết lưu text, CTC evidence và metadata biến đổi. **Không sử dụng GT, ID ảnh hoặc tên dataset tại inference.**

```text
Image I → Frozen SVTRv2-T → original hypothesis h0
        → candidate generator (orientation / Geometry v3 / width / overlap crop)
        → contextual evidence z(I, reference, candidate, other candidates)
        → Orientation Arbiter / Recovery Arbiter
        → utility & calibrated acceptance → final transcription
```

### 2.1. Candidate generation — giữ crop

Trong `sage_gpu_adapter.py`, ảnh được cắt trên **raw PIL image trước resize**. `longtext_final.py` tạo các window dùng aspect ratio `6.0`, overlap `0.5`. `longtext_v3.py` trước tiên dùng overlap-matching, sau đó dùng ước lượng tọa độ x của CTC events khi overlap matching thất bại. Nếu số window trên ảnh gốc `n_orig > 1`, hệ thống sinh `geometry_v3`. Với ảnh cao-hẹp (`W <= H`), sinh cả `rotate_cw` và `rotate_ccw`. Với ảnh ngang mà `n_orig == 1`, sinh `crop3_width3` (cùng cấu hình Unified v1); nếu dự đoán gốc dài >=8 ký tự, còn sinh `crop3_msr` và `crop3_width2`. Nếu dự đoán gốc dài >=8 **hoặc** CTC pressure >=0.4, và ảnh ngang, sinh `full_width2`.

**Crop không bị bỏ.** Khác với Unified v1, sinh candidate không có nghĩa candidate đó được chọn. Tuy nhiên notebook hiện vẫn chạy một số candidate nặng ngay trên ảnh từ ngắn; chưa thể claim tiết kiệm inference GPU.

### 2.2. Reference-aware hypothesis

`reference(obs)` ưu tiên candidate `geometry_v3` **nếu candidate này xuất hiện trong `obs.alternatives`**, nếu không dùng `original`. GPU adapter chỉ đưa geometry vào alternatives khi nó được tạo và khác bản nguyên ảnh. Với reference Geometry, bản nguyên vẫn được đưa vào candidates để bộ chọn có thể quay lại. **Reference không phải bảo đảm đúng.**

### 2.3. Contextual evidence

Với reference `r`, candidate `c` và tập ứng viên `H`, bộ feature `z_c=phi(I,h0,r,c,H)` gồm một biến categorical (`strategy`) và **36 biến số**: aspect, độ dài, pressure/entropy/edge occupancy của bản gốc; confidence, CTC margin, overlap agreement của candidate; mức tương đồng chuỗi và chênh lệch với reference; thông tin chiều xoay cạnh tranh; cùng các dấu hiệu mất ký tự. Source: `src/sage_str_v4/selector.py` (`NUMERIC`, `CATEGORICAL`, `features`). CTC pressure nhiều crop **không được dùng như tổng frame để so trực tiếp với một chuỗi full-image**; feature chính là `original_slot_pressure`.

### 2.4. Contextual Dual-Arbiter

`Orientation Arbiter` xét `rotate_cw/rotate_ccw`. `Recovery Arbiter` xét bản nguyên thay reference Geometry, tăng width và các candidate crop. Hai head là logistic regression đa lớp, với nhãn đối chứng dựa trên **reference**:

- **Gain:** candidate đúng toàn dòng và reference sai.
- **Harm:** reference đúng toàn dòng và candidate sai.
- **Neutral:** cả hai cùng đúng hoặc cả hai cùng sai.

Ba lớp **loại trừ nhau**; tổng xác suất bằng một. Song song, Ridge regression ước lượng thay đổi normalized character edit error `ΔCER=(ED(reference,GT)-ED(candidate,GT))/len(GT)` trong huấn luyện. `Temperature scaling` trên split `calibration` hiệu chỉnh các xác suất nhóm. Khác với việc fine-tune SVTRv2, các bộ chọn nhỏ được huấn luyện trên nhãn, còn backbone frozen.

### 2.5. Quyết định rủi ro

Code v4 hiện dùng:

`U(c) = p_gain(c) - 2.5·p_harm(c) + predicted_ΔCER(c) - 0.002·fixed_cost(c)`

và ngưỡng nhóm `τ_orientation=0.65`, `τ_recovery=-0.10` ở **pilot cụ thể**. Chỉ chọn ứng viên đủ điều kiện có utility cao nhất; nếu không, giữ reference. Các trọng số và chi phí quy ước là **thiết kế thực nghiệm**, chưa có bảo đảm rủi ro theo lý thuyết. Không dùng wall-clock GPU làm feature quyết định. Mô hình chỉ được hiệu chuẩn/điều chỉnh trước khi đánh giá eval.

## 3. Vì sao các thành phần cùng phục vụ một nguyên lý?

`Crop/stitching`, `rotation`, `width` là **cơ chế sinh giả thuyết**, không phải ba đóng góp độc lập. `Contextual feature` đánh giá một giả thuyết theo **tình trạng và các phương án cạnh tranh**. `Dual-Arbiter` mô hình hóa hai loại failure mode, trong khi `Gain/Harm/Neutral + CER utility` thể hiện bài toán **chọn thay thế rủi ro**. Đây là một phương pháp ra quyết định nhất quán, không chỉ một tập lệnh `if` áp dụng cứng mọi biến đổi.

## 4. Thuật toán tóm tắt

```text
Input: image I, frozen SVTRv2-T, trained Orientation/Recovery selector
h0 ← recognize(I)
H ← generate_candidates(I, h0) using same frozen recognizer
r ← geometry_v3 if its distinct proposal exists else h0
for c in H \ {r}:
    z_c ← contextual_evidence(I, h0, r, c, H)
    (pG, pH, pN), ΔCER_hat ← evaluate_with_head(c.group, z_c)
    U_c ← pG - 2.5*pH + ΔCER_hat - 0.002*fixed_cost(c)
choose eligible c* with maximum U_c above group threshold
return c*.text if one exists, otherwise r.text
```

Lưu ý: chi tiết tie-break, audit fallback và chuẩn hóa text theo đúng code được lưu trong notebook và source, không thay thuật toán bằng pseudocode.

## 5. Điều chưa được chứng minh

- Không có guarantee không regression, không có conformal risk control và không xác nhận calibration OOD.
- `Contextual` vs no-context, `Dual` vs single-head, CER utility vs exact-only và chính sách reference chưa được ablation đầy đủ trên cùng giao thức.
- Vì candidate tính trước, chưa thể claim nhanh/tiết kiệm VRAM hơn SVTRv2-T.
- Full cohort và untouched external test chưa hoàn tất; calibration Orientation trong pilot chỉ có 27 candidate records.
- Một số ảnh LTB vẫn bị rút ngắn nghiêm trọng; SAGE v4 không nhất quán tốt hơn v3 locked ở CER.

## 6. Kết luận vị trí đóng góp

**Đóng góp kỹ thuật:** một *contextual risk-aware hypothesis selection framework* cho frozen scene-text recognizer, không phải phát minh mới về crop, rotation hay CTC. Báo cáo đồ án có thể mô tả phương pháp như một hướng thực nghiệm có hệ thống, nhưng không nên tuyên bố SOTA/generalization đã xác nhận.
