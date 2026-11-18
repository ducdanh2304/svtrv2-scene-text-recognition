# SAGE-STR v4 — Báo cáo nghiên cứu tổng hợp (CS406)

**Phương pháp đề xuất:** Contextual Risk-Aware Adaptive Inference for Scene Text Recognition  
**Backbone:** Official SVTRv2-T — frozen  
**Phiên bản code:** SAGE-STR v4, đúng notebook Kaggle và selector SHA256 `f5ff65b03239c56cd2a7fb5ab39262bde8e10ee86dc25116e9713cf6f94e1c55`  
**Phạm vi kết quả:** Development pilot GPU, không phải đánh giá độc lập trên phân phối mới.

## Tóm tắt

Bài toán trọng tâm là nhận dạng cảnh văn bản có độ dài và hướng đọc biến thiên mà không gây hồi quy lớn ở văn bản thông thường. Sau chuỗi thử nghiệm tăng MSR, crop/stitching, hợp nhất logits, Geometry v3, Unified v1, ablation tác động crop/resize và các phiên bản SAGE v2–v4, nghiên cứu đề xuất **lựa chọn giả thuyết nhận dạng dựa trên bằng chứng ngữ cảnh và rủi ro**. Cùng một SVTRv2-T cố định tạo ra các ứng viên nguyên ảnh, xoay, tăng resolution và crop chồng lấn. Bộ chọn có học ước lượng Gain/Harm/Neutral và lợi ích giảm lỗi ký tự, sau đó quyết định thay dự đoán tham chiếu hay giữ lại.

**Quan sát trên pilot eval (461 ảnh):** LTB 52,70%, Multi-Oriented 86,21%, Multi-Words 85,00%, IIIT5K 97,87%, CUTE80 96,36% exact-line accuracy. Đây là các tỷ lệ trên từng eval subset, không phải điểm benchmark đầy đủ. V4 chưa có bảo đảm không regression, chưa chứng minh SOTA/tổng quát hóa, và vẫn cần ablation từng thành phần. Đặc biệt, LTB CER v4 9,706% kém v3 locked 9,330% trên cùng cohort, và mức vượt Unified trên LTB/Multi-Oriented chưa có ý nghĩa thống kê (McNemar p=0,375).

## Mục lục

1. Hành trình nghiên cứu và các thử nghiệm thất bại
2. Kiến trúc SAGE-STR v4, các thành phần và hàm utility
3. Kết quả GPU, phép kiểm tra paired và giới hạn claim
4. Hướng dẫn tái lập và protocol đánh giá
5. Nguồn tham khảo, attribution và dữ liệu

---


## Phần 1: 01 HANH TRINH THU NGHIEM

**Phạm vi:** Tổng hợp các thí nghiệm đã thực hiện, kết quả có thể kiểm tra lại từ output, nguyên nhân chọn/bỏ từng hướng, và thiết kế phương pháp hiện tại. Các kết quả được phân biệt theo backbone, tập ảnh và giao thức; **không coi các lần thử trên LTB/Union14M/IIIT5K/CUTE80 là kiểm chứng tổng quát hóa độc lập**.

## 1. Bài toán và giả thuyết

SVTRv2-T là recognizer theo CTC. Trên dòng dài, việc resize nguyên ảnh vào chiều rộng giới hạn có thể làm chữ bị nén, thiếu timestep/chi tiết và tăng **deletion**. Chia ảnh trước khi resize giúp đọc nhiều ký tự hơn, nhưng nếu chia một từ ngắn thành ba mảnh thì ngữ cảnh và hình dạng ký tự bị phá; xoay ảnh không cần thiết cũng gây regression.

**Giả thuyết cuối:** Không có một phép biến đổi luôn tốt. Cần sinh nhiều giả thuyết nhận dạng với cùng backbone, rồi lựa chọn theo bằng chứng và rủi ro *trên từng ảnh*, có quyền giữ reference.

## 2. Toàn bộ chuỗi thí nghiệm và quyết định

| Giai đoạn | Câu hỏi/thiết kế | Quan sát thực nghiệm | Quyết định |
|---|---|---|---|
| **A. MSR nguyên dòng (lịch sử SVTRv2-B)** | Tăng `max_ratio` để bớt nén chữ | LTB A-Avg **19,56%** (ratio12) → **30,29%** (≥64); nhóm 56+ chỉ **12,78%** | Chỉ tăng width không giải quyết hết chữ rất dài. **Lưu ý: backbone B, không so trực tiếp với T.** |
| **B. Crop + Stitch v1/v2 (lịch sử B)** | Cắt raw image trước MSR, overlap để bảo toàn ký tự | Stitch v1 dễ lặp ký tự; Stitch v2 (Levenshtein/geometry) đạt **48,89% A-Avg** theo best-of-8 trên LTB, vượt resize nguyên ảnh | **Giữ crop**. Bản gốc chọn tỷ lệ cửa sổ 6, overlap 50%; đây là tối ưu exploratory trên LTB, không phải hằng số lý thuyết. |
| **C. Posterior fusion / phase / context diagnostics** | Ghép CTC logits bằng center, uniform, taper; đổi phase, phân tích crop edge | Uniform giảm điểm mạnh (766 ảnh: A-Avg **25,59%**), center **45,51%**, taper **46,14%** (chênh không thuyết phục); synthetic tests chỉ ra ảnh hưởng ngữ cảnh/position | Không dùng average logits như giải pháp chính; lỗi không chỉ do seam. |
| **D. Learned fusion: LAG, GROW/POINTWISE, GAWS** | Huấn luyện tầng fusion/CTC với backbone frozen | GROW/POINTWISE sau sửa lỗi nhãn case-sensitive: trên 300 LTB A-Avg **47,33/48,00%** vs geometry **47,00%**; chênh không chắc chắn. GAWS 3.352 ảnh: frozen + beam/blank khoảng **49,01% A-Avg**, head-only **49,25%**, nhưng preprocessing và đối chứng v2 chưa tương đương | Không chọn learned fusion làm solution chính vì thiếu chứng minh vượt đối chứng crop chuẩn; không quảng bá là chưa từng huấn luyện. |
| **E. Stitch v2 → Geometry v3, checkpoint T** | Khi overlap match thất bại, dùng vị trí event CTC nguyên tọa độ ảnh để cắt seam | **Full 4.789 LTB**, v2 A-Avg **45,72%**, exact overall **50,07%**, CER **14,73%**; Geometry v3 **47,00%**, **50,68%**, **14,53%** | **Giữ Geometry v3** làm phép ghép cho các cửa sổ. |
| **F. BGAS/ECO seam refinements** | Blank-guided seam, event consensus/arbitration | BGAS báo cáo smoke chỉ **4 ảnh** (không thể claim full từ smoke). ECO **4.789 ảnh**: event_consensus giữ A-Avg **47,00%**; CER chỉ nhích **14,5264 → 14,5234%**, không tăng exact. `gap_support` A-Avg **47,13%** nhưng biên nhỏ | Không thêm các tầng seam phức tạp vào final; lợi ích nhỏ/thiếu chắc chắn. |
| **G. Unified v1 (T, full LTB)** | Orientation nếu ảnh cao; Dense forced 3 crops ×3 nếu một cửa sổ; còn lại Geometry v3 | **4.789 LTB:** A-Avg **47,00→49,61%** (so với v3), overall exact **50,68→55,13%**, CER **14,53→8,30%**. Orientation cứu 108, Dense net +105, v3 giữ 3.966 ảnh | Có lợi rõ trên *LTB exploratory*, nhưng route được thiết kế quá cứng. |
| **H. External regression** | Kiểm tra bốn bộ chữ thường / đa hướng | Unified v1 so với SVTRv2-T gốc: IIIT5K **98,60→77,13%**; CUTE80 **96,52→66,55%**; Multi-Words **82,63→76,84%**; Multi-Oriented **75,68→75,60%** | **Bác bỏ Unified v1 làm pipeline tổng quát**; không thể đem nó ra ứng dụng cho mọi ảnh. |
| **I. Root-cause ablation trên cùng ảnh** | Tách ảnh hưởng crop, width ×2/×3, stitching; 128 ảnh/dataset | IIIT5K 127/128 baseline, **122/128** khi ép 3 crop thường, **105/128** khi thêm width×3. Multi-Oriented **86→13→8**. CUTE80 **123→111→84**. V3 stitch thường sửa một phần lỗi so v2 | Lỗi chính do **router chọn can thiệp sai, crop phá ngữ cảnh, width×3 méo glyph**; không nên chỉ sửa stitching. |
| **J. SAGE v2 — selective fallback** | Original làm reference; selector dự đoán gain/harm và được quyền giữ gốc | Pilot eval 448 ảnh: bảo vệ IIIT5K/CUTE80 **51/52** đúng mỗi tập; Multi-Oriented **35/52** vs Unified **42/52**; LTB exact **128/240** nhưng CER xấu (**~12,75%** vs Unified **~8,70%**) | Fallback original quá thường xuyên trên chữ dài; orientation bị từ chối; phải sửa mô hình rủi ro. |
| **K. TRACE v2 forensic** | Ghi toàn bộ điểm ứng viên | Phát hiện `p_gain+p_harm>1` ở nhiều candidate vì hai mô hình độc lập; utility phụ thuộc GPU timing; CTC T của crop overlap không so trực tiếp với full image | Sửa bằng phân loại **Gain/Harm/Neutral** nhất quán, fixed cost, feature crop-normalized. |
| **L. SAGE v3 — Dual-Arbiter** | Hai head Orientation/Recovery; CER-aware; geometry-preferred reference | Pilot eval: LTB **127/240**, CER **8,480%** vs Unified **128/240**, CER **8,686%**; Multi-Oriented **38/52** vs Unified **42/52**; IIIT5K/CUTE80 giữ baseline | Cơ chế hợp lý hơn nhưng selection orientation còn yếu, vẫn có regression. |
| **M. SAGE v4 — contextual selection** | Thêm thông tin ứng viên cạnh tranh, locked hash split, fit v3/v4 trên cùng GPU predictions | **Eval 461 ảnh:** LTB **127/241 (52,70%)**, CER **9,71%**; Multi-Oriented **50/58 (86,21%)**; Multi-Words **51/60 (85,00%)**; IIIT5K **46/47 (97,87%)**; CUTE80 **53/55 (96,36%)** | **Chốt làm phương pháp đề xuất trong đồ án**, không gọi là validated SOTA hay hết regression. |

**Chú ý phân biệt cohort:** Full 4.789 LTB (v3/Unified) là thử nghiệm khác với eval 241 ảnh của v4. A-Avg v4 trên 241 ảnh = **44,37%**; Unified v1 trên đúng eval đó = **43,37%**. Trong cùng lượt v4, v3_locked cũng **127/241** nhưng CER **9,33%** tốt hơn v4 **9,71%**.

## 3. Method cuối — ngắn gọn, đúng triển khai

**SAGE-STR v4 = frozen SVTRv2-T + candidate hypotheses + contextual dual-arbiter.**

1. **Đọc gốc:** SVTRv2-T nhận dạng ảnh nguyên, xuất chuỗi và bằng chứng CTC.
2. **Sinh ứng viên:** tùy hình học/CTC, có thể xoay 90° CW/CCW, dùng Geometry v3 nhiều window, tăng full width ×2, hoặc **forced 3 overlapping crops** với MSR hay width ×2/×3. **Crop không bị bỏ.** Với Geometry, cắt ảnh gốc *trước* khi từng crop được resize, tỷ lệ cửa sổ 6 và overlap 50%; v3 ghép kết quả theo overlap matching và original-coordinate seam khi cần. Hiện mã vẫn sinh crop3×3 trên nhiều ảnh một-window (kể cả từ ngắn) để bộ chọn học từ chối; vì vậy chưa thể claim tiết kiệm GPU.
3. **Chọn reference:** ưu tiên dự đoán Geometry v3 nếu nó khác bản gốc và có nhiều window; nếu không dùng dự đoán gốc. Reference không phải đáp án bảo đảm đúng; gốc vẫn có thể trở thành ứng viên thay thế.
4. **Trích evidence:** dùng bằng chứng CTC, độ dài, sự đồng thuận/chênh giữa reference, original, crop và hai chiều xoay; không nhìn GT khi suy luận.
5. **Hai bộ chọn học được:** Orientation head và Recovery head; logistic regression dự đoán ba trạng thái **Gain/Harm/Neutral** (loại trừ nhau), Ridge dự đoán lợi ích giảm lỗi ký tự. Học train, temperature calibration trên calib, threshold trên tune, audit và eval tách theo hash định danh.
6. **Ra quyết định:** chọn candidate đủ điều kiện có điểm tiện ích cao nhất: `U = P(gain) - 2.5 P(harm) + estimated CER gain - 0.002 fixed cost`; nếu không vượt ngưỡng, giữ reference. `τ_orientation=0.65`, `τ_recovery=-0.10` là các ngưỡng thực nghiệm v4 pilot, **không phải hằng số phổ quát**.

### Vì sao đây là một method thống nhất?

Một cơ chế xuyên suốt: **đề xuất nhiều cách đọc cùng một ảnh → ước lượng lợi ích và nguy cơ thay kết quả → lựa chọn hoặc từ chối.** Xoay/crop/resize chỉ là cách tạo giả thuyết, không phải ba đóng góp tách rời. Phần đóng góp đề xuất là *contextual risk-aware hypothesis selection*; backbone vẫn frozen nhưng selector **có học giám sát**, nên **không claim toàn bộ pipeline training-free**.

## 4. Điều gì claim được và chưa được?

**Có thể nói:** Trên các thí nghiệm phát triển với SVTRv2-T frozen, phép xử lý crop/Geometry giúp phục hồi chuỗi dài; SAGE v4 hạn chế đáng kể regression chữ ngắn do Unified v1 và đạt cải thiện observed exact accuracy so với SVTRv2-T gốc trên LTB/Multi-Oriented. Đã có ablation cùng ảnh, lịch sử thử sai và nguyên nhân kỹ thuật cho kiến trúc hiện tại.

**Không thể nói:** Cropping bị loại bỏ; mọi thành phần đều đã có ablation độc lập; v4 vượt v3 ở mọi metric; không còn regression; chắc chắn tổng quát hóa ra mọi ảnh; safe theo nghĩa có bảo đảm; nhanh hơn SVTRv2-T; hay phương pháp có novelty/SOTA đã được kiểm chứng trước literature.

**Kiểm chứng còn thiếu:** ablation một head vs hai head, bỏ contextual features, bỏ CER reward, thay reference; full benchmark + latency; chạy lại seed; test độc lập **chưa sử dụng trong quá trình thiết kế** (đặc biệt chữ dài ≥26). Khác biệt v4–Unified trong LTB eval và Multi-Oriented eval chỉ **+3 ảnh mỗi tập**, McNemar exact `p=0.375` mỗi tập: chưa đủ khẳng định vượt trội thống kê.

## 5. Kết luận

Giữ **crop (trong candidate pool)**, giữ **Geometry v3 (stitching)**, **không giữ forced crop như quyết định bắt buộc**, và dùng một bộ chọn thích ứng thống nhất. Có thể báo cáo **SAGE-STR v4 là phương pháp đề xuất của đồ án**, với kết quả pilot minh bạch và giới hạn rõ. Để claim tổng quát hóa mạnh hoặc nghiên cứu học thuật, phải hoàn thành ablation, full test và independent held-out evaluation.

### Nguồn kết quả đã kiểm tra (file gốc)

- `CS431_CS406_MASTER_HANDOFF_2026-11-17.md` — lịch sử các nhánh MSR, SAGE-CTC, GROW, GAWS, v2.
- `ECO_LTB_RESULTS.zip` (`ECO_RESULTS.md`, `summary.csv`) — 4.789 ảnh, Stitch v2/v3/blank/event ablation.
- `CS406_UNIFIED_LTB_RESULTS.zip` — 4.789 ảnh, full route and matched comparison.
- `CS406_EXTERNAL_SVTRv2_vs_UNIFIED_RESULTS.zip` — external datasets regression.
- `CS406_DENSE_ROOT_CAUSE_RESULTS.zip` — crop/resolution/stitch ablation.
- `SAGE_STR_V2_GPU_RESULTS.zip`, `SAGE_STR_V2_GPU_RESULTS (1).zip` — selective fallback và TRACE.
- `SAGE_STR_V4_GPU_RESULTS.zip` — Kaggle v4, source code, eval matched, audit, paired tests.

---


## Phần 2: 02 PHUONG PHAP CHI TIET

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

---


## Phần 3: 03 KET QUA VA GIOI HAN

**Nguồn:** ZIP Kaggle `SAGE_STR_V4_GPU_RESULTS.zip` tải lên sau khi chạy GPU; số liệu dưới đây chép từ CSV **trên các image IDs giống nhau trong lượt v4**. Không thay đổi code để chạy lại checkpoint ở môi trường đóng gói.

## 1. Thiết lập kiểm chứng

- **Official frozen SVTRv2-T**, OpenOCR commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- CKPT SHA256 `684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6`; CONFIG SHA256 `f1ab35a0f5e44d3c24c940cdf1a07404068ad9191bad9fe232ad4fcf38bbbe29`.
- LTB 1.200 ảnh và mỗi external dataset tối đa 256 ảnh ở pilot: **2.224 ảnh inference**. Có **461 ảnh eval hợp lệ**; còn lại gồm train/calibration/tune/audit. Mốc hash salt `cs406-sage-v4-locked-splits-v1` không đổi theo cap.
- GT chỉ dùng ở huấn luyện selector và chấm hậu nghiệm, không đưa vào candidate generation/selection khi inference.
- **Toàn bộ nguồn benchmark đã từng được khảo sát trong quá trình phát triển**, nên eval chỉ là *development evaluation*, chưa phải untouched generalization.

## 2. Accuracy và CER trên eval

| Dataset (n) | SVTRv2-T gốc | Unified v1 | SAGE v3 cùng split | SAGE v4 | CER v4 |
|---|---:|---:|---:|---:|---:|
| LTB (241) | 33,20% | 51,45% | 52,70% | **52,70%** | 9,706% |
| Multi-Oriented (58) | 72,41% | 81,03% | 84,48% | **86,21%** | 3,951% |
| Multi-Words (60) | 81,67% | 76,67% | 83,33% | **85,00%** | 4,810% |
| IIIT5K (47) | **97,87%** | 80,85% | **97,87%** | **97,87%** | 0,417% |
| CUTE80 (55) | 92,73% | 60,00% | 94,55% | **96,36%** | 0,738% |

LTB A-Avg: **44,366%** v4, **43,373%** Unified v1, **18,802%** original SVTRv2-T. LTB CER v4 **9,706%** = Unified v1 **9,706%**, còn v3 locked **9,330%** (**tốt hơn v4**). Accuracy v4 và v3 locked đều **127/241**, nhưng các chuỗi dự đoán không nhất thiết trùng nhau.

**LTB bucket:** 26–35: **92/172**; 36–55: **32/55**; 56+: **3/14**. Đặc biệt nhóm 56+ quá nhỏ và accuracy thấp để claim giải quyết văn bản cực dài.

## 3. Kết quả paired (cùng ảnh)

| So sánh (v4 so với Unified v1) | Win | Loss | Net | McNemar exact p |
|---|---:|---:|---:|---:|
| LTB | 4 | 1 | +3 | 0,375 |
| Multi-Oriented | 4 | 1 | +3 | 0,375 |
| Multi-Words | 6 | 1 | +5 | 0,125 |
| IIIT5K | 8 | 0 | +8 | 0,0078125 |
| CUTE80 | 20 | 0 | +20 | 0,0000019 |

Vì p=0,375 trên LTB và Multi-Oriented, **không thể claim vượt Unified ổn định có ý nghĩa thống kê** trong hai tập này. So v4 với SVTRv2-T original trên LTB có **53 win/6 loss**, không có bảo đảm zero harm.

## 4. Audit và reproducibility

Pilot v4 `audit` trên **306 ảnh**: **20 wins, 2 harms**, với `edit_gain=366`. Chính sách pass kiểm tra phát triển trong notebook, không có bảo đảm formal risk/OOD. Split sizes: `train 1003, calibration 230, tune 224, audit 306, eval 461`. Code selector SHA256 `f5ff65b03239c56cd2a7fb5ab39262bde8e10ee86dc25116e9713cf6f94e1c55` đúng với source từ ZIP GPU.

Các file có thể tái tính thống kê: `results/evaluation_predictions.csv`, `evaluation_accuracy_cer_by_dataset.csv`, `evaluation_paired_win_loss.csv`, `ltb_eval_length_buckets.csv` và `ltb_eval_AAvg.csv`. Run `python experiments/long_text/sage_str_v4/scripts/verify_pilot_metrics.py` tại root.

## 5. Điều được claim / không được claim

**Có thể claim có điều kiện:** (i) đóng góp thiết kế selection theo risk/context trên cùng frozen SVTRv2-T, (ii) trong **pilot**, SAGE hạn chế phần lớn regression chữ ngắn của Unified v1 và cải thiện exact-match trên LTB/Multi-Oriented so với original, (iii) Geometry crop giúp long text nhưng ép crop không thích hợp có thể gây hại.

**Không được claim:** phương pháp hoàn thiện tuyệt đối, zero regression, SOTA, verified external generalization, training-free, hiệu quả GPU, conformal guarantee hoặc tất cả component đã được ablation chứng minh độc lập. `source-aware` constraints chỉ dùng cho tuning/evaluation ở phát triển; inference không đọc tên dataset.

## 6. Việc còn lại để chốt nghiên cứu mạnh hơn

1. Đóng băng mã + selector + split, kiểm tra reproducibility nhiều lần/seed trên cùng candidate cache.
2. Chạy **full 4.789 LTB + full external** (báo cáo tách pilot và full).
3. Ablation one-head vs dual; context vs no-context; CER-aware vs exact-only; reference original vs Geometry; từng candidate family trên cùng cache.
4. Đánh giá trên benchmark long-text và short-text **chưa từng dùng để phân tích/chọn phương pháp**; báo cáo latency/VRAM/cost.

---


## Phần 4: 04 TAI LAP VA KIEM CHUNG

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

---


## Phần 5: 05 NGUON VA TRICH DAN

- **SVTRv2:** Yongkun Du et al., *SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition*, ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html
- **Official OpenOCR:** https://github.com/Topdu/OpenOCR, pinned commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`. Original author licenses apply to external checkout.
- **CTC:** Graves et al., *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*, ICML 2006. https://doi.org/10.1145/1143844.1143891
- **Dataset source:** https://huggingface.co/datasets/topdu/OpenOCR-Data; the Kaggle notebook pins revision `d59c13643f1d7f92831625fe161e2c19ec3d7f52` and uses LTB + Union14M / IIIT5K / CUTE80 components. Respect original dataset licenses.
- **Local research provenance:** experiments/methods described in `01_HANH_TRINH_THU_NGHIEM.md` with historical CSVs and previous notebooks in the CS406 research archive.

Do not claim independent novelty of cropping, rotation, stitching, CTC, MSR or the backbone. SAGE-STR's proposed technical contribution is the **risk-aware contextual selection framework** combining these as candidate generators on a frozen backbone. No formal publication or peer review of the method has been asserted.

---
