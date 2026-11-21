# Đếm số mẫu SỬA ĐÚNG / LÀM SAI THÊM của FUSE so với baseline B0 (theo quy tắc repo nhóm).
# Chạy trên Kaggle sau khi Add Input dataset trctrnthanh386/fuse-release (không cần GPU).
import csv, glob, re, os
cands = glob.glob('/kaggle/input/**/preds_fuse_release', recursive=True) + glob.glob('/kaggle/input/**/fuse_release/preds', recursive=True)
PRED = next(p for p in cands if glob.glob(f'{p}/*__FUSE.tsv'))
def norm(s): return re.sub(r'[^a-z0-9]', '', (s or '').lower())
def load(s, m): return {r['id']: r for r in csv.DictReader(open(f'{PRED}/{s}__{m}.tsv', encoding='utf-8'), delimiter='\t')}
print('preds:', PRED)
for s in ['union_art', 'iiit5k_test', 'wordart_bench']:
    for base in ['B0_goc', 'WATERec_RS_goc']:
        B, F = load(s, base), load(s, 'FUSE'); assert B.keys() == F.keys()
        ok = lambda P, i: norm(P[i]['pred']) == norm(P[i]['gt'])
        fix = sum(ok(F, i) and not ok(B, i) for i in B); brk = sum(ok(B, i) and not ok(F, i) for i in B)
        print(f'{s:14s} FUSE vs {base:15s}: n={len(B)} | sửa đúng {fix} | làm sai thêm {brk} | ròng {fix - brk:+d}')
