# CS406 — Từ nhận dạng chữ dài đến SAGE-STR v4: báo cáo quá trình nghiên cứu

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
