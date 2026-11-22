# Attribution, nguồn gốc mã và trích dẫn

- **SVTRv2:** Yongkun Du et al., *SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition*, ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html
- **Official OpenOCR:** https://github.com/Topdu/OpenOCR, pinned commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`. Original author licenses apply to external checkout.
- **CTC:** Graves et al., *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*, ICML 2006. https://doi.org/10.1145/1143844.1143891
- **Dataset source:** https://huggingface.co/datasets/topdu/OpenOCR-Data; the Kaggle notebook pins revision `d59c13643f1d7f92831625fe161e2c19ec3d7f52` and uses LTB + Union14M / IIIT5K / CUTE80 components. Respect original dataset licenses.
- **Local research provenance:** experiments/methods described in `01_HANH_TRINH_THU_NGHIEM.md` with historical CSVs and previous notebooks in the CS406 research archive.

Do not claim independent novelty of cropping, rotation, stitching, CTC, MSR or the backbone. SAGE-STR's proposed technical contribution is the **risk-aware contextual selection framework** combining these as candidate generators on a frozen backbone. No formal publication or peer review of the method has been asserted.
