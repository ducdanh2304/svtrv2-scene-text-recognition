# ==================================================================================
#  GAC (Glyph-Anchored Consistency) — 1 CELL ĐỘC LẬP (Kaggle, GPU T4 x2, Internet ON)
#  Phần 1: setup + dữ liệu + hàm dùng chung (lấy từ cell pipeline) + train R3_lowlr làm đối chứng (bỏ qua nếu đã có)
#  Phần 2: train GAC@R3_lowlr (giống hệt R3, chỉ thêm loss GAC) -> đánh giá -> so sánh có CI95
#  ---- phần dưới là mô tả gốc của cell pipeline ----
#  B1 evaluator đủ ID + CER thật | B2 fine-tune B0 trên WATER-T + WATER-R + IIIT5K
#  B3 biến thể: skel / rope / dcn  | B4 ensemble (chọn luật trên val)
#  Chạy lại cell an toàn: bước nào đã xong (có file DONE) sẽ được bỏ qua.
# ==================================================================================
import os, sys, io, re, csv, json, glob, time, shutil, random, hashlib, subprocess, copy, builtins
_T0 = time.time()
def print(*a, **k):   # in kèm thời gian đã chạy + flush ngay (thấy tiến độ cả ở chế độ Commit)
    k.setdefault('flush', True); builtins.print(f'[{(time.time() - _T0) / 60:6.1f}m]', *a, **k)
print('Bắt đầu cell')

# ----------------------------- CẤU HÌNH (chỉ sửa ở đây) ---------------------------
RUN = dict(
    setup=True,        # cài gói, clone repo, sitecustomize
    data=True,         # tải tập test + WATER-T/WATER-R, tạo LMDB train/val cho từng tỉ lệ dữ liệu
    eval_base=False,    # đánh giá B0 gốc + WATERec-RS gốc (zero-shot)
    recipes=['R3_lowlr'],   # đối chứng của GAC: train lại CHỈ khi chưa có runs/R3_lowlr/DONE
    skel=False,        # + skeleton head   } chạy trên BASE_RECIPE
    rope=False,        # + 2D-RoPE  (cần patches/rope_patch.py)
    dcn=False,         # + DCN      (cần patches/dcn_patch.py)
    ensemble=False,    # bật khi đã có >= 2 model fine-tune
)
# Mỗi recipe chỉ khác recipe đứng trước MỘT yếu tố.
#   data : tỉ lệ dữ liệu (xem COMPOSITIONS) | val: 'mix' = val lấy từ data train, 'wordart' = WordArt-Bench làm dev
#   freeze: số stage đầu của encoder bị đóng băng (0 = train hết)
RECIPES = {
    'B0_art':    dict(data='T50k_R20k', lr=5e-5, freeze=0, val='mix'),       # đã chạy: Union -3.89
    'R1_valWA':  dict(data='T50k_R20k', lr=5e-5, freeze=0, val='wordart'),   # đổi: chọn checkpoint bằng WordArt-Bench
    'R2_ratio':  dict(data='T20k_R50k', lr=5e-5, freeze=0, val='wordart'),   # đổi: tỉ lệ tổng hợp:thật ~ 2:5
    'R3_lowlr':  dict(data='T20k_R50k', lr=1e-5, freeze=0, val='wordart'),   # đổi: LR thấp hơn
    'R4_freeze': dict(data='T20k_R50k', lr=5e-5, freeze=2, val='wordart'),   # đổi: đóng băng Stage 1-2
}
# BOOST (không train thêm, chọn mọi thứ trên DEV = WordArt-Bench, test = Union ART):
#   wise: nội suy trọng số B0_goc <-> run fine-tune tốt nhất trên dev (WiSE-FT)
#   tta : test-time augmentation (khung resize lân cận, xoay ảnh dọc) — chỉ giữ nếu dev tăng
#   fuse: kết hợp hệ SVTRv2 tốt nhất với WATERec theo confidence, ngưỡng δ chọn trên dev
BOOST = dict(wise=False, tta=False, fuse=False)
WISE_ALPHAS = [0.2, 0.35, 0.5, 0.65, 0.8]
BASE_RECIPE   = 'R2_ratio'   # recipe nền cho skel/rope/dcn — CHỈ đổi sau khi xem kết quả recipe trên WordArt-Bench (dev)
COMPOSITIONS  = {'T50k_R20k': (50000, 20000), 'T20k_R50k': (20000, 50000)}   # (WATER-T, WATER-R)
EPOCHS        = 8          # số epoch fine-tune mỗi biến thể
BS_PER_GPU    = 64
IIIT_REPEAT   = 3          # nhân bản IIIT5K train (chữ thường) trong tập train
N_VAL_T, N_VAL_R = 500, 500
SKEL_LAMBDA   = 0.1
SKEL_STAGE    = 1          # 0-based: 1 = Stage 2 của SVTRv2
SEED          = 0

W     = '/kaggle/working/art_pipeline'
REPO  = '/kaggle/working/OpenOCR-WATERec'
RAW   = '/tmp/water_raw'                       # file tải lớn để ngoài /kaggle/working (giới hạn 20GB)
DATA  = f'{W}/data'
PRED  = f'{W}/preds'
B0_CKPT  = '/kaggle/input/notebooks/trctrnthanh386/xulianh-deblur/restormer_truc/official_svtrv2_base/best.pth'
IIIT_DIR = '/kaggle/input/datasets/prathmeshzade/iiit5k-words'
for d in [W, DATA, PRED, f'{W}/patches', f'{W}/runs']: os.makedirs(d, exist_ok=True)
random.seed(SEED)

def sh(cmd, check=True):
    print('$', cmd if isinstance(cmd, str) else ' '.join(cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if r.returncode and check:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise RuntimeError(f'Lệnh lỗi: {cmd}')
    return r

# ================================== B0. SETUP =====================================
if RUN['setup']:
    sh(f'{sys.executable} -m pip install -q lmdb rapidfuzz pyclipper shapely')
    sh(f'{sys.executable} -m pip install -q --no-deps imgaug')
    if not os.path.isdir(REPO):
        sh(f'git clone -q --depth 1 https://github.com/YesianRohn/OpenOCR-WATERec {REPO}')
    os.makedirs(f'{W}/compat', exist_ok=True)
    with open(f'{W}/compat/sitecustomize.py', 'w') as f:
        f.write(
            "import numpy as np\n"
            "if not hasattr(np, 'sctypes'):\n"
            "    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64],\n"
            "                  'uint': [np.uint8, np.uint16, np.uint32, np.uint64],\n"
            "                  'float': [np.float16, np.float32, np.float64],\n"
            "                  'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}\n")
    import torch
    print('GPU:', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print('Dung lượng trống /tmp: %.1f GB | /kaggle/working: %.1f GB' % (
        shutil.disk_usage('/tmp').free / 2**30, shutil.disk_usage('/kaggle/working').free / 2**30))
    assert os.path.exists(B0_CKPT), f'Không thấy checkpoint B0: {B0_CKPT} -> Add Input notebook chứa nó'
for p in [REPO, f'{W}/compat']:
    if p not in sys.path: sys.path.insert(0, p)
sys.path.insert(0, W)
exec(open(f'{W}/compat/sitecustomize.py').read())   # vá np.sctypes cả trong kernel hiện tại
os.chdir(REPO)

# ============================ CORE (dùng chung mọi bước) ==========================
import lmdb, numpy as np, yaml, torch
from PIL import Image

DICT_PATH = f'{REPO}/tools/utils/EN_symbol_dict.txt'
CHARSET = set(open(DICT_PATH, encoding='utf-8').read().split('\n')) - {''}

def lmdb_count(path):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as t: return int(t.get(b'num-samples'))

def lmdb_iter(path, idx=None):
    """yield (i, img_bytes, label)."""
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as t:
        n = int(t.get(b'num-samples'))
        for i in (idx if idx is not None else range(1, n + 1)):
            ib, lb = t.get(b'image-%09d' % i), t.get(b'label-%09d' % i)
            if ib is not None and lb is not None: yield i, bytes(ib), lb.decode('utf-8')

def lmdb_write(path, samples):
    """samples: list[(img_bytes, label)] -> LMDB chuẩn OpenOCR (+ key wh để RatioDataSet đọc nhanh)."""
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)
    env = lmdb.open(path, map_size=64 * 2**30); j = 0
    with env.begin(write=True) as t:
        for ib, lb in samples:
            try: w, h = Image.open(io.BytesIO(ib)).size
            except Exception: continue
            j += 1
            t.put(b'image-%09d' % j, ib); t.put(b'label-%09d' % j, lb.encode('utf-8'))
            t.put(b'wh-%09d' % j, f'{w}_{h}'.encode())
        t.put(b'num-samples', str(j).encode())
    env.close(); print(f'  ghi {path}: {j} mẫu'); return j

def label_ok(lb): return 1 <= len(lb) <= 25 and all(c in CHARSET for c in lb)
def md5(b): return hashlib.md5(b).hexdigest()
def norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())   # = RecMetric(is_filter, is_lower, ignore_space)

def lev(a, b):
    if a == b: return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

# ---- file variants.py: dùng cho cả train (subprocess DDP) lẫn eval ----
VARIANTS_SRC = r'''
import os, types, importlib.util, torch, torch.nn as nn, torch.nn.functional as F
PATCH_DIR = os.environ.get('ART_PATCH_DIR', '/kaggle/working/art_pipeline/patches')
LAMBDA = float(os.environ.get('ART_SKEL_LAMBDA', '0.1'))
STAGE = int(os.environ.get('ART_SKEL_STAGE', '1'))
_S = {}

def _load_patch(name):
    p = os.path.join(PATCH_DIR, f'{name}_patch.py')
    if not os.path.exists(p):
        raise FileNotFoundError(f'Thiếu {p}: cần hàm apply(model) -> model')
    spec = importlib.util.spec_from_file_location(f'{name}_patch', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

class SkeletonHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Upsample(scale_factor=(4, 4), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.Conv2d(64, 1, 1))
    def forward(self, x): return self.net(x)

def _to2d(out):
    x, sz = out
    if x.dim() == 3:
        B, N, C = x.shape; H, Wd = int(sz[0]), int(sz[1])
        x = x.transpose(1, 2).reshape(B, C, H, Wd)
    return x

def add_skeleton(model):
    model.encoder.stages[STAGE].register_forward_hook(lambda m, i, o: _S.__setitem__('feat', _to2d(o)))
    was = model.training; model.eval()
    with torch.no_grad(): model.encoder(torch.zeros(1, 3, 32, 128))
    c = _S.pop('feat').shape[1]; model.train(was)
    model.skel_head = SkeletonHead(c)
    orig = model.forward
    def fwd(self, x, data=None):
        out = orig(x, data=data)
        if self.training:
            _S['logit'] = self.skel_head(_S.pop('feat')); _S['img'] = x
        return out
    model.forward = types.MethodType(fwd, model)
    print(f'[skel] head trên encoder.stages[{STAGE}], C={c}, lambda={LAMBDA}')
    return model

def freeze_stages(model, k):
    mods = [getattr(model.encoder, 'pope', None)] + list(model.encoder.stages[:k])
    n = 0
    for m in mods:
        if m is None: continue
        for p in m.parameters(): p.requires_grad = False; n += p.numel()
    print(f'[freeze] patch-embed + {k} stage đầu: {n / 1e6:.2f}M tham số bị đóng băng')

def apply(model, variant, train):
    if variant in ('rope', 'dcn'):
        model = _load_patch(variant).apply(model) or model
    if variant == 'skel' and train:
        model = add_skeleton(model)
    k = int(os.environ.get('ART_FREEZE', '0'))
    if train and k > 0:
        freeze_stages(model, k)
    return model

# ---- pseudo-skeleton: Otsu + soft-skeleton (clDice) trên chính ảnh đã augment ----
def _otsu(flat, bins=64):
    B = flat.shape[0]
    x = ((flat.clamp(-1, 1) + 1) / 2 * (bins - 1)).round().long()
    hist = torch.zeros(B, bins, device=flat.device).scatter_add_(1, x, torch.ones_like(flat))
    p = hist / hist.sum(1, keepdim=True); w0 = p.cumsum(1)
    mu = (p * torch.arange(bins, device=flat.device)).cumsum(1); muT = mu[:, -1:]
    sb = (muT * w0 - mu) ** 2 / (w0 * (1 - w0) + 1e-8)
    return sb.argmax(1).float() / (bins - 1) * 2 - 1

def _erode(x): return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))
def _dilate(x): return F.max_pool2d(x, 3, 1, 1)
def _soft_skel(x, iters=5):
    skel = F.relu(x - _dilate(_erode(x)))
    for _ in range(iters):
        x = _erode(x); d = F.relu(x - _dilate(_erode(x))); skel = skel + F.relu(d - skel * d)
    return skel

@torch.no_grad()
def pseudo_skeleton(img, out_hw):
    g = img.float().mean(1, keepdim=True)
    B = g.shape[0]; t = _otsu(g.view(B, -1)).view(B, 1, 1, 1)
    border = torch.cat([g[:, :, 0, :], g[:, :, -1, :]], -1).flatten(1).mean(1).view(B, 1, 1, 1)
    fg = torch.where(border > t, (g < t).float(), (g >= t).float())
    sk = _soft_skel(fg)
    return (F.adaptive_max_pool2d(sk, tuple(out_hw)) > 0.5).float()

class SkelLoss(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base
    def forward(self, preds, batch):
        d = self.base(preds, batch)
        logit, img = _S.pop('logit', None), _S.pop('img', None)
        if logit is None: return d
        tgt = pseudo_skeleton(img, logit.shape[-2:]); l = logit.float()
        frac = tgt.mean((1, 2, 3)); ok = (frac > 0.02) & (frac < 0.20)   # bỏ mẫu nền texture làm skeleton nhiễu
        if ok.sum() == 0:
            d['loss'] = d['loss'] + 0 * l.sum(); return d
        l, tgt = l[ok], tgt[ok]
        bce = F.binary_cross_entropy_with_logits(l, tgt, pos_weight=torch.tensor(5., device=l.device))
        p = l.sigmoid(); dice = 1 - (2 * (p * tgt).sum() + 1) / (p.sum() + tgt.sum() + 1)
        sl = bce + dice
        d['loss'] = d['loss'] + LAMBDA * sl; d['skel_loss'] = sl.detach()
        return d

def wrap_loss(loss, variant):
    return SkelLoss(loss) if variant == 'skel' else loss
'''
with open(f'{W}/variants.py', 'w', encoding='utf-8') as f: f.write(VARIANTS_SRC)

TRAIN_SRC = r'''
import os, sys, runpy
REPO = os.environ['ART_REPO']; sys.path.insert(0, REPO); sys.path.insert(0, os.environ['ART_W']); os.chdir(REPO)
import variants, openrec.modeling as M, openrec.losses as L
V = os.environ.get('ART_VARIANT', 'base')
_bm, _bl = M.build_model, L.build_loss
M.build_model = lambda cfg: variants.apply(_bm(cfg), V, train=True)
L.build_loss = lambda cfg: variants.wrap_loss(_bl(cfg), V)
sys.argv = ['tools/train_rec.py'] + sys.argv[1:]
runpy.run_path('tools/train_rec.py', run_name='__main__')
'''
with open(f'{W}/train_variant.py', 'w', encoding='utf-8') as f: f.write(TRAIN_SRC)
import importlib; import variants; importlib.reload(variants)

# ------------------------------ evaluator từng ảnh -------------------------------
from torchvision import transforms as T
from torchvision.transforms import functional as TF
_TO_T = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

RENAMED = {'CTCDecoder2D': 'RCTCDecoder'}   # tên cũ (OpenOCR 2024) -> tên hiện tại, code giống hệt

def _rename(o):
    if isinstance(o, dict): return {k: _rename(v) for k, v in o.items()}
    if isinstance(o, list): return [_rename(v) for v in o]
    return RENAMED.get(o, o) if isinstance(o, str) else o

def _buildable(cfg):
    from openrec.modeling import build_model
    from openrec.losses import build_loss
    from openrec.postprocess import build_post_process
    c = copy.deepcopy(cfg)
    post = build_post_process(c['PostProcess'], c['Global'])
    c['Architecture']['Decoder']['out_channels'] = post.get_character_num()
    build_model(c['Architecture']); build_loss(c['Loss'])

_SVTR_CFG_CACHE = {}
def svtr_cfg(ckpt=None):
    if ckpt in _SVTR_CFG_CACHE: return copy.deepcopy(_SVTR_CFG_CACHE[ckpt])
    base = yaml.safe_load(open(f'{REPO}/configs/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml', encoding='utf-8'))
    base['Global']['character_dict_path'] = DICT_PATH; base['PostProcess']['character_dict_path'] = DICT_PATH
    cfg = base
    if ckpt:
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        if isinstance(ck, dict) and isinstance(ck.get('config'), dict) and 'Architecture' in ck['config']:
            cand = copy.deepcopy(base)
            for k in ['Architecture', 'Loss', 'PostProcess']:
                if k in ck['config']: cand[k] = _rename(copy.deepcopy(ck['config'][k]))
            if 'character_dict_path' in cand['PostProcess']: cand['PostProcess']['character_dict_path'] = DICT_PATH
            try:
                _buildable(cand); cfg = cand
                print('  dùng Architecture/Loss/PostProcess lưu trong checkpoint (đã đổi tên module cũ)')
            except Exception as e:
                print(f'  config trong checkpoint không build được ({e}); dùng svtrv2_smtr_gtc_rctc.yml')
    _SVTR_CFG_CACHE[ckpt] = copy.deepcopy(cfg)
    return cfg

def waterec_cfg():
    cfg = yaml.safe_load(open(f'{REPO}/configs/rec/waterec/navit_ar.yml', encoding='utf-8'))
    cfg['Global']['character_dict_path'] = DICT_PATH; cfg['PostProcess']['character_dict_path'] = DICT_PATH
    return cfg

class Recognizer:
    def __init__(self, cfg, ckpt, variant='base', device='cuda', tta=False):
        self.tta = tta
        from openrec.modeling import build_model
        from openrec.postprocess import build_post_process
        cfg = copy.deepcopy(cfg)
        self.post = build_post_process(cfg['PostProcess'], cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels'] = self.post.get_character_num()
        self.model = variants.apply(build_model(cfg['Architecture']), variant, train=False)
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items() if 'skel_head' not in k}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing: raise RuntimeError(f'{len(missing)} key THIẾU khi load {ckpt}: {missing[:8]}')
        if unexpected: print(f'  [cảnh báo] {len(unexpected)} key thừa: {unexpected[:5]}')
        self.dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.dev).eval()
        tr = [list(o)[0] for o in cfg['Eval']['dataset']['transforms']]
        self.navit = 'NaRecTVResize' in tr
        if self.navit:
            from openrec.preprocess.resize import NaRecTVResize
            a = [o for o in cfg['Eval']['dataset']['transforms'] if 'NaRecTVResize' in o][0]['NaRecTVResize']
            self.na = NaRecTVResize(**a)
        else:   # giống RatioDataSetTVResize ở chế độ Eval (padding=False)
            self.max_ratio = cfg['Eval']['loader'].get('max_ratio', 12)
            self.base_shape = cfg['Eval']['dataset'].get('base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
            self.base_h = cfg['Eval']['dataset'].get('base_h', 32)

    def prep(self, img, r=None):
        if self.navit: return self.na({'image': img})['image']
        w, h = img.size
        if r is None: r = int(np.clip(np.around(w / h), 1, self.max_ratio))
        iw, ih = self.base_shape[r - 1] if r <= 4 else [self.base_h * r, self.base_h]
        return _TO_T(TF.resize(img, (ih, iw), interpolation=T.InterpolationMode.BICUBIC))

    @torch.no_grad()
    def _run(self, img, r=None):
        x = self.prep(img, r).to(self.dev)
        preds = self.model([x]) if self.navit else self.model(x[None])
        out = self.post(preds)
        if len(out) == 2 and isinstance(out[0], list):           # GTC: [gtc, ctc] -> chính = CTC (= 'acc' OpenOCR)
            (tg, sg), (tc, sc) = out[0][0], out[1][0]
            return tc, float(sc), tg
        t, s = out[0]; return t, float(s), ''

    def __call__(self, ib):
        img = Image.open(io.BytesIO(ib)).convert('RGB')
        best = self._run(img)
        if not self.tta: return best
        # TTA: thử thêm khung resize lân cận (SVTRv2) + xoay ±90° cho ảnh dọc; giữ kết quả có confidence cao nhất
        cands = []
        w, h = img.size
        if not self.navit:
            r0 = int(np.clip(np.around(w / h), 1, self.max_ratio))
            cands += [(img, r) for r in (r0 - 1, r0 + 1) if 1 <= r <= self.max_ratio]
        if h > 1.3 * w:
            cands += [(img.rotate(90, expand=True), None), (img.rotate(-90, expand=True), None)]
        for im, r in cands:
            o = self._run(im, r)
            if o[1] > best[1]: best = o
        return best

TESTSETS = {}   # tên -> đường dẫn LMDB
def register_tests():
    for name, p in [('union_art', f'{DATA}/u14m/artistic'), ('wordart_bench', f'{DATA}/WordArt-Bench'),
                    ('iiit5k_test', f'{DATA}/iiit5k_test'), ('val', f'{DATA}/val')]:
        if os.path.exists(f'{p}/data.mdb'): TESTSETS[name] = p

def predict(name, cfg, ckpt, variant='base', sets=None, tta=False):
    sets = sets or list(TESTSETS)
    todo = [s for s in sets if not os.path.exists(f'{PRED}/{s}__{name}.tsv')]
    if not todo: print(f'[pred] {name}: đã có'); return
    rec = Recognizer(cfg, ckpt, variant, tta=tta); print(f'[pred] {name} ({variant}{", TTA" if tta else ""}) ckpt={ckpt}')
    for s in todo:
        t0 = time.time(); rows = []
        for i, ib, lb in lmdb_iter(TESTSETS[s]):
            p, sc, pg = rec(ib); rows.append([f'{i:09d}', lb, p, f'{sc:.5f}', pg])
            if len(rows) % 500 == 0: print(f'   {s}: {len(rows)} ảnh...')
        with open(f'{PRED}/{s}__{name}.tsv', 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f, delimiter='\t'); wr.writerow(['id', 'gt', 'pred', 'score', 'pred_gtc']); wr.writerows(rows)
        print(f'   {s}: {len(rows)} ảnh, {time.time() - t0:.0f}s')
    del rec; torch.cuda.empty_cache()

def load_pred(s, name):
    return {r['id']: r for r in csv.DictReader(open(f'{PRED}/{s}__{name}.tsv', encoding='utf-8'), delimiter='\t')}

def score(s, name, col='pred'):
    P = load_pred(s, name); ids = sorted(P)
    n_lmdb = lmdb_count(TESTSETS[s])
    assert len(ids) == n_lmdb, f'{s}/{name}: {len(ids)} dự đoán khác {n_lmdb} mẫu LMDB'
    G = [norm(P[i]['gt']) for i in ids]; Q = [norm(P[i][col]) for i in ids]
    c = np.array([a == b for a, b in zip(Q, G)]); e = np.array([lev(a, b) for a, b in zip(Q, G)])
    L = np.array([len(g) for g in G])
    return ids, c, e, L

def paired_ci(s, a, b, B=10000):
    _, ca, ea, L = score(s, a); _, cb, eb, _ = score(s, b)
    rng = np.random.default_rng(SEED); da, dc = [], []
    for _ in range(B):
        k = rng.integers(0, len(L), len(L))
        da.append((cb[k].mean() - ca[k].mean()) * 100); dc.append((eb[k].sum() - ea[k].sum()) / max(L[k].sum(), 1) * 100)
    return ((cb.mean() - ca.mean()) * 100, np.percentile(da, 2.5), np.percentile(da, 97.5),
            (eb.sum() - ea.sum()) / L.sum() * 100, np.percentile(dc, 2.5), np.percentile(dc, 97.5))

def report(pairs=()):
    names = sorted({os.path.basename(p).split('__')[1][:-4] for p in glob.glob(f'{PRED}/*__*.tsv')})
    rows = []
    print('\n================ KẾT QUẢ (chuẩn hoá: lowercase + [a-z0-9]; CER = tổng edit / tổng ký tự GT) ================')
    for s in TESTSETS:
        for m in names:
            if not os.path.exists(f'{PRED}/{s}__{m}.tsv'): continue
            ids, c, e, L = score(s, m)
            r = dict(set=s, model=m, n=len(ids), acc=round(c.mean() * 100, 4), cer=round(e.sum() / L.sum() * 100, 4))
            if any(load_pred(s, m)[i]['pred_gtc'] for i in ids[:5]):
                _, cg, eg, _ = score(s, m, 'pred_gtc'); r['gtc_acc'] = round(cg.mean() * 100, 4)
            rows.append(r); print(r)
    with open(f'{W}/summary.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['set', 'model', 'n', 'acc', 'cer', 'gtc_acc']); wr.writeheader(); wr.writerows(rows)
    for s in ['union_art', 'wordart_bench', 'iiit5k_test']:
        for a, b in pairs:
            if s in TESTSETS and all(os.path.exists(f'{PRED}/{s}__{x}.tsv') for x in (a, b)):
                d = paired_ci(s, a, b)
                print(f'[{s}] {b} - {a}: Δacc {d[0]:+.3f} CI95[{d[1]:+.3f},{d[2]:+.3f}] | ΔCER {d[3]:+.3f} CI95[{d[4]:+.3f},{d[5]:+.3f}]')
    print(f'-> {W}/summary.csv')

# ===================================== B1+B2. DATA ================================
def hf(repo, fn, local):
    from huggingface_hub import hf_hub_download
    print(f'  tải {repo}/{fn} -> {local} (file lớn có thể mất 5-20 phút, không có thanh tiến độ)')
    return hf_hub_download(repo_id=repo, filename=fn, repo_type='dataset', local_dir=local)

def comp_dir(tag):
    # bản B0_art cũ ghi thẳng vào DATA/train, DATA/val -> giữ nguyên để không phải tạo lại
    if tag == 'T50k_R20k' and os.path.exists(f'{DATA}/train/data.mdb') and os.path.exists(f'{DATA}/DONE'): return DATA
    return f'{DATA}/{tag}'

def needed_recipes():
    rs = list(RUN['recipes'])
    if any(RUN[v] for v in ('skel', 'rope', 'dcn')): rs.append(BASE_RECIPE)
    return [r for r in dict.fromkeys(rs)]

if RUN['data']:
    # --- tập test (dùng chung) ---
    if not os.path.exists(f'{DATA}/u14m/artistic/data.mdb'):
        for fn in ['u14m/artistic/data.mdb', 'u14m/artistic/lock.mdb']: hf('topdu/OpenOCR-Data', fn, DATA)
    if not os.path.exists(f'{DATA}/WordArt-Bench/data.mdb'):
        for fn in ['WordArt-Bench/data.mdb', 'WordArt-Bench/lock.mdb']: hf('Yesianrohn/WATER-Data', fn, DATA)
    print('Union ART num-samples =', lmdb_count(f'{DATA}/u14m/artistic'), '| WordArt-Bench =', lmdb_count(f'{DATA}/WordArt-Bench'))

    # --- IIIT5K: test -> đối chứng chữ thường; train -> iiit_train / iiit_val (10%) ---
    if not (os.path.exists(f'{DATA}/iiit5k_test/data.mdb') and os.path.exists(f'{DATA}/iiit_train/data.mdb')):
        root = glob.glob(f'{IIIT_DIR}/**/IIIT5K', recursive=True)
        root = [r for r in root if os.path.isdir(f'{r}/train')][0]
        def iiit_csv(kind):
            c = glob.glob(f'{IIIT_DIR}/**/{kind}data.csv', recursive=True); assert c, f'không thấy {kind}data.csv'
            out = []
            for r in csv.DictReader(open(c[0], encoding='utf-8')):
                with open(os.path.join(root, r['ImgName']), 'rb') as f: out.append((f.read(), r['GroundTruth'].strip()))
            return out
        if not os.path.exists(f'{DATA}/iiit5k_test/data.mdb'): lmdb_write(f'{DATA}/iiit5k_test', iiit_csv('test'))
        tr = iiit_csv('train'); random.Random(SEED).shuffle(tr); n_v = len(tr) // 10
        lmdb_write(f'{DATA}/iiit_val', [s for s in tr[:n_v] if label_ok(s[1])])
        lmdb_write(f'{DATA}/iiit_train', [s for s in tr[n_v:] if label_ok(s[1])])

    # --- WordArt-Bench bản lọc nhãn hợp lệ: CHỈ dùng làm dev để chọn checkpoint lúc train ---
    if not os.path.exists(f'{DATA}/wordart_dev/data.mdb'):
        lmdb_write(f'{DATA}/wordart_dev', [(ib, lb) for _, ib, lb in lmdb_iter(f'{DATA}/WordArt-Bench') if label_ok(lb)])

    # --- tập train/val cho từng tỉ lệ dữ liệu cần dùng ---
    tags = list(dict.fromkeys(RECIPES[r]['data'] for r in needed_recipes()))
    todo = [t for t in tags if not os.path.exists(f'{comp_dir(t)}/train/data.mdb')]
    if todo:
        ban = {md5(ib) for p in [f'{DATA}/u14m/artistic', f'{DATA}/WordArt-Bench', f'{DATA}/iiit5k_test'] for _, ib, _ in lmdb_iter(p)}
        print('md5 bị chặn (trùng tập test):', len(ban))
        def sample(path, k, rng):
            if k <= 0: return []
            n = lmdb_count(path); idx = sorted(rng.sample(range(1, n + 1), min(n, int(k * 1.3))))
            out = [(ib, lb) for _, ib, lb in lmdb_iter(path, idx) if label_ok(lb) and md5(ib) not in ban]
            rng.shuffle(out); out = out[:k]
            print(f'  lấy {len(out)}/{k} từ {path} (tổng {n})'); return out
        for fn in ['WATER-S/WATER-T/data.mdb', 'WATER-S/WATER-T/lock.mdb', 'WATER-R/data.mdb', 'WATER-R/lock.mdb']:
            if not os.path.exists(f'{RAW}/{fn}'): hf('Yesianrohn/WATER-Data', fn, RAW)
        iiit_tr = [(ib, lb) for _, ib, lb in lmdb_iter(f'{DATA}/iiit_train')]
        iiit_val = [(ib, lb) for _, ib, lb in lmdb_iter(f'{DATA}/iiit_val')]
        for t in todo:
            nt, nr = COMPOSITIONS[t]; rng = random.Random(f'{SEED}-{t}'); d = comp_dir(t)
            wt = sample(f'{RAW}/WATER-S/WATER-T', nt + N_VAL_T, rng)
            wr_ = sample(f'{RAW}/WATER-R', nr + N_VAL_R, rng)
            train = wt[N_VAL_T:] + wr_[N_VAL_R:] + iiit_tr * IIIT_REPEAT; rng.shuffle(train)
            val = wt[:N_VAL_T] + wr_[:N_VAL_R] + iiit_val
            lmdb_write(f'{d}/train', train); lmdb_write(f'{d}/val', val)
            json.dump(dict(water_t=len(wt) - N_VAL_T, water_r=len(wr_) - N_VAL_R, iiit_train=len(iiit_tr),
                           iiit_repeat=IIIT_REPEAT, val=len(val), seed=SEED), open(f'{d}/composition.json', 'w'), indent=1)
register_tests(); print('Tập đánh giá:', {k: lmdb_count(v) for k, v in TESTSETS.items()})

# ============================ B1. ĐÁNH GIÁ MODEL GỐC ==============================
if RUN['eval_base']:
    from huggingface_hub import hf_hub_download
    wck = hf_hub_download('Yesianrohn/WATERec-Models', 'WATERec-RS.pth', local_dir=f'{W}/ckpt')
    predict('WATERec_RS_goc', waterec_cfg(), wck)
    predict('B0_goc', svtr_cfg(B0_CKPT), B0_CKPT)
    report()

# ======================== B2/B3. FINE-TUNE (B0_art, skel, rope, dcn) ==============
def make_train_cfg(name, recipe):
    cfg = svtr_cfg(B0_CKPT)
    out = f'{W}/runs/{name}'
    d = comp_dir(recipe['data'])
    val_dir = f'{DATA}/wordart_dev' if recipe['val'] == 'wordart' else f'{d}/val'
    g = cfg['Global']
    g.update(epoch_num=EPOCHS, output_dir=out, pretrained_model=B0_CKPT, checkpoints=None,
             eval_batch_step=[0, 500], eval_epoch_step=[0, 1], save_epoch_step=[EPOCHS + 1, 1],
             save_res_path=f'{out}/predicts.txt', use_amp=True, print_batch_step=50)
    if os.path.exists(f'{out}/latest.pth') and not os.path.exists(f'{out}/DONE'):
        g['checkpoints'] = f'{out}/latest.pth'; g['pretrained_model'] = None; print('  resume từ latest.pth')
    cfg['Optimizer']['lr'] = recipe['lr']
    cfg['LRScheduler'] = dict(name='OneCycleLR', warmup_epoch=0.5, cycle_momentum=False)
    for mode, p_ in [('Train', f'{d}/train'), ('Eval', val_dir)]:
        assert os.path.exists(f'{p_}/data.mdb'), f'thiếu {p_} -> bật RUN["data"]'
        cfg[mode]['dataset']['data_dir_list'] = [p_]
        cfg[mode]['sampler']['first_bs'] = BS_PER_GPU
        cfg[mode]['loader']['batch_size_per_card'] = BS_PER_GPU
        cfg[mode]['loader']['num_workers'] = 2
    os.makedirs(out, exist_ok=True)
    p = f'{out}/config.yml'; yaml.safe_dump(cfg, open(p, 'w', encoding='utf-8'), allow_unicode=True, sort_keys=False)
    json.dump(recipe, open(f'{out}/recipe.json', 'w'), indent=1)
    return p, out

def train(name, recipe, variant='base'):
    out = f'{W}/runs/{name}'
    if os.path.exists(f'{out}/DONE'): print(f'[train] {name}: đã xong'); return out
    p, out = make_train_cfg(name, recipe)
    print(f'[train] {name} | variant={variant} | recipe={recipe}')
    env = dict(os.environ, ART_REPO=REPO, ART_W=W, ART_VARIANT=variant, ART_PATCH_DIR=f'{W}/patches',
               ART_SKEL_LAMBDA=str(SKEL_LAMBDA), ART_SKEL_STAGE=str(SKEL_STAGE), ART_FREEZE=str(recipe['freeze']),
               PYTHONPATH=f'{W}/compat:{W}:{REPO}:' + os.environ.get('PYTHONPATH', ''))
    ngpu = max(1, torch.cuda.device_count())
    cmd = [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={ngpu}', '--master_port=29511',
           f'{W}/train_variant.py', '--c', p]
    print(f'  {" ".join(cmd)}\n  log: {out}/train.log')
    t0 = time.time(); keep = re.compile(r'best metric|cur metric|epoch: \[\d+/\d+\].*global_step: \d*[05]00\b|Error|Traceback|skel|freeze|finetune|resume')
    with open(f'{out}/train.log', 'w') as lf:
        pr = subprocess.Popen(cmd, env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in pr.stdout:
            lf.write(line)
            if keep.search(line): print('  ', line.rstrip()[:300])
        pr.wait()
    if pr.returncode != 0:
        print(open(f'{out}/train.log').read()[-4000:]); raise RuntimeError(f'train {name} lỗi (xem {out}/train.log)')
    open(f'{out}/DONE', 'w').write(f'{time.time() - t0:.0f}s'); print(f'[train] {name} xong sau {(time.time() - t0) / 3600:.2f} giờ')
    return out

JOBS = [(r, RECIPES[r], 'base') for r in RUN['recipes']]
for v in ['skel', 'rope', 'dcn']:
    if not RUN[v]: continue
    if v in ('rope', 'dcn') and not os.path.exists(f'{W}/patches/{v}_patch.py'):
        print(f'[bỏ qua {v}] chưa có {W}/patches/{v}_patch.py'); continue
    JOBS.append((f'{v}@{BASE_RECIPE}', RECIPES[BASE_RECIPE], v))
for name, recipe, v in JOBS:
    out = train(name, recipe, v)
    predict(name, svtr_cfg(B0_CKPT), f'{out}/best.pth', variant=(v if v in ('rope', 'dcn') else 'base'))

# ============================ B5. BOOST (không train thêm) ========================
DEV = 'wordart_bench'
BOOST_PAIRS = []

def ckpt_of(name):
    if name == 'B0_goc': return B0_CKPT
    if name == 'WATERec_RS_goc': return f'{W}/ckpt/WATERec-RS.pth'
    if name.startswith('WiSE_'): return f'{W}/runs/{name}.pth'
    return f'{W}/runs/{name}/best.pth'

def cfg_of(name): return waterec_cfg() if name.startswith('WATERec') else svtr_cfg(B0_CKPT)

def dev_acc(name):
    if not os.path.exists(f'{PRED}/{DEV}__{name}.tsv'): return -1.0
    _, c, e, L = score(DEV, name); return float(c.mean() - 1e-6 * e.sum() / L.sum())   # hoà acc -> CER thấp hơn

def load_sd(p):
    ck = torch.load(p, map_location='cpu', weights_only=False); sd = ck.get('state_dict', ck)
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items() if 'skel_head' not in k}

if any(BOOST.values()) and not os.path.exists(ckpt_of('WATERec_RS_goc')):
    from huggingface_hub import hf_hub_download
    hf_hub_download('Yesianrohn/WATERec-Models', 'WATERec-RS.pth', local_dir=f'{W}/ckpt')

best_svtr = 'B0_goc'
if BOOST['wise']:
    ft_runs = [r for r in RECIPES if os.path.exists(f'{W}/runs/{r}/best.pth') and os.path.exists(f'{PRED}/{DEV}__{r}.tsv')]
    if ft_runs:
        ft = max(ft_runs, key=dev_acc); print(f'\n[WiSE] nội suy B0_goc <-> {ft} (run fine-tune tốt nhất trên dev)')
        s0, s1 = load_sd(B0_CKPT), load_sd(f'{W}/runs/{ft}/best.pth')
        assert set(s0) == set(s1), f'key khác nhau: {list(set(s0) ^ set(s1))[:5]}'
        for a in WISE_ALPHAS:
            name = f'WiSE_{ft}_a{a}'
            if not os.path.exists(ckpt_of(name)):
                sd = {k: ((1 - a) * s0[k].float() + a * s1[k].float()).to(s1[k].dtype) if s1[k].is_floating_point() else s1[k]
                      for k in s1}
                torch.save({'state_dict': sd}, ckpt_of(name))
            predict(name, cfg_of(name), ckpt_of(name), sets=[DEV])
        del s0, s1
        cand = ['B0_goc', ft] + [f'WiSE_{ft}_a{a}' for a in WISE_ALPHAS]
        print('  dev acc:', {c: round(dev_acc(c) * 100, 3) for c in cand})
        best_svtr = max(cand, key=dev_acc); print('  -> chọn', best_svtr)
        predict(best_svtr, cfg_of(best_svtr), ckpt_of(best_svtr))
        BOOST_PAIRS.append(('B0_goc', best_svtr))
    else:
        print('[WiSE] chưa có run fine-tune nào có dự đoán trên dev -> bỏ qua')

best_sys = {'svtr': best_svtr, 'waterec': 'WATERec_RS_goc'}
if BOOST['tta']:
    for key, base in best_sys.items():
        name = base + '+TTA'
        predict(name, cfg_of(base), ckpt_of(base), sets=[DEV], tta=True)
        print(f'[TTA] {base}: dev {dev_acc(base) * 100:.3f} -> +TTA {dev_acc(name) * 100:.3f}')
        if dev_acc(name) > dev_acc(base):
            predict(name, cfg_of(base), ckpt_of(base), tta=True); best_sys[key] = name
            BOOST_PAIRS.append((base, name))

if BOOST['fuse']:
    A, B = best_sys['svtr'], best_sys['waterec']
    if all(os.path.exists(f'{PRED}/{s}__{m}.tsv') for s in TESTSETS for m in (A, B)):
        def fuse_rows(s, d):
            PA, PB = load_pred(s, A), load_pred(s, B); rows = []
            for i in sorted(PA):
                a, b = PA[i], PB[i]
                pick = b if (norm(a['pred']) == norm(b['pred']) or float(b['score']) + d >= float(a['score'])) else a
                rows.append([i, a['gt'], pick['pred'], pick['score'], ''])
            return rows
        def rows_acc(rows): return np.mean([norm(r[1]) == norm(r[2]) for r in rows])
        grid = [round(x, 2) for x in np.arange(-0.5, 0.51, 0.05)]
        dev_scores = {d: rows_acc(fuse_rows(DEV, d)) for d in grid}
        d_best = max(grid, key=lambda d: (dev_scores[d], -abs(d)))
        print(f'[FUSE] {A} + {B}: δ={d_best} (dev acc {dev_scores[d_best] * 100:.3f}; '
              f'chỉ {B}: {dev_acc(B) * 100:.3f}, chỉ {A}: {dev_acc(A) * 100:.3f})')
        for s in TESTSETS:
            with open(f'{PRED}/{s}__FUSE.tsv', 'w', newline='', encoding='utf-8') as f:
                wr = csv.writer(f, delimiter='\t'); wr.writerow(['id', 'gt', 'pred', 'score', 'pred_gtc'])
                wr.writerows(fuse_rows(s, d_best))
        json.dump(dict(A=A, B=B, delta=d_best, dev=DEV), open(f'{W}/fuse.json', 'w'), indent=1)
        BOOST_PAIRS += [('B0_goc', 'FUSE'), ('WATERec_RS_goc', 'FUSE')]
    else:
        print(f'[FUSE] thiếu dự đoán của {A} hoặc {B} trên đủ các tập -> bỏ qua')

# ================================ B4. ENSEMBLE ====================================
SELECT_SET = 'wordart_bench'   # luật ensemble chọn trên dev (WordArt-Bench), không dùng Union
if RUN['ensemble']:
    cands = [n for n, _, _ in JOBS] + list(RECIPES) + ['WATERec_RS_goc']
    members = [m for m in dict.fromkeys(cands) if all(os.path.exists(f'{PRED}/{s}__{m}.tsv') for s in TESTSETS)]
    assert len(members) >= 2, f'cần >=2 model có dự đoán trên mọi tập, hiện: {members}'
    val_acc = {m: score(SELECT_SET, m)[1].mean() for m in members}
    order = sorted(members, key=lambda m: -val_acc[m]); print(f'ưu tiên ({SELECT_SET} acc):', {m: round(val_acc[m] * 100, 2) for m in order})
    for s in TESTSETS:
        P = {m: load_pred(s, m) for m in members}; ids = sorted(P[members[0]]); rows = []
        for i in ids:
            votes = {}
            for m in order: votes.setdefault(norm(P[m][i]['pred']), []).append(m)
            best = max(votes.items(), key=lambda kv: (len(kv[1]), -order.index(kv[1][0])))
            rows.append([i, P[members[0]][i]['gt'], P[best[1][0]][i]['pred'], '', ''])
        with open(f'{PRED}/{s}__ENS.tsv', 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f, delimiter='\t'); wr.writerow(['id', 'gt', 'pred', 'score', 'pred_gtc']); wr.writerows(rows)

# ================================ PHẦN 2. GAC ======================================
GAC_ALPHA = 0.3        # 1 giá trị duy nhất (không dò), vừa phải để không lấn loss gốc
GAC_STAGE = 2          # 0-based: stage cuối của encoder SVTRv2
GAC_COLS  = 8
BASE      = 'R3_lowlr'
NAME      = f'GAC@{BASE}'
_d = comp_dir(RECIPES[BASE]['data'])
for _p in [f'{_d}/train', f'{DATA}/wordart_dev', f'{DATA}/u14m/artistic']:
    assert os.path.exists(f'{_p}/data.mdb'), f'thiếu {_p} -> kiểm tra RUN["data"]=True và recipes=["{BASE}"] ở đầu cell'

GAC_SRC = r'''
import os, types, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
ALPHA = float(os.environ['GAC_ALPHA']); STAGE = int(os.environ['GAC_STAGE']); COLS = int(os.environ['GAC_COLS'])
_S = {}
_FONTS = [p for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                      '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'] if os.path.exists(p)]
_CACHE = {}

def _to2d(o):
    x, sz = o if isinstance(o, (tuple, list)) else (o, None)
    if x.dim() == 3:
        B, N, C = x.shape; x = x.transpose(1, 2).reshape(B, C, int(sz[0]), int(sz[1]))
    return x

def render(text, h, w):
    """Chữ đen nền trắng bằng font thường, căng vừa khung h x w (giống ảnh crop chữ). -> tensor [-1,1]."""
    k = (text, h, w)
    if k in _CACHE: return _CACHE[k]
    fs = 64
    font = ImageFont.truetype(_FONTS[0], fs) if _FONTS else ImageFont.load_default(size=fs)
    l, t, r, b = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), text, font=font)
    im = Image.new('L', (max(1, r - l) + 16, max(1, b - t) + 16), 255)
    ImageDraw.Draw(im).text((8 - l, 8 - t), text, font=font, fill=0)
    im = im.resize((w, h), Image.BICUBIC).convert('RGB')
    x = torch.from_numpy(__import__('numpy').asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1
    if len(_CACHE) < 200000: _CACHE[k] = x
    return x

def _pool(f):   # [B,C,H,W] -> [B,C,COLS], chuẩn hoá theo kênh
    return F.normalize(F.adaptive_avg_pool2d(f.float(), (1, COLS)).flatten(2), dim=1)

def apply(model, cfg, build, b0_ckpt, chars):
    teacher = build(cfg)
    ck = torch.load(b0_ckpt, map_location='cpu', weights_only=False); sd = ck.get('state_dict', ck)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    miss, _ = teacher.load_state_dict(sd, strict=False); assert not miss, f'mỏ neo thiếu key: {miss[:5]}'
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    object.__setattr__(model, '_gac_teacher', teacher)        # KHÔNG đăng ký submodule -> không vào DDP/checkpoint
    model.encoder.stages[STAGE].register_forward_hook(lambda m, i, o: _S.__setitem__('s', _to2d(o)) if m.training else None)
    teacher.encoder.stages[STAGE].register_forward_hook(lambda m, i, o: _S.__setitem__('t', _to2d(o)))
    _S['chars'] = chars
    orig = model.forward
    def fwd(self, x, data=None):
        out = orig(x, data=data)
        if self.training: _S['img'] = x
        return out
    model.forward = types.MethodType(fwd, model)
    print(f'[GAC] mỏ neo = B0 đóng băng ({b0_ckpt}), stage {STAGE}, {COLS} cột, alpha={ALPHA}, font={_FONTS[:1] or "PIL mặc định"}')
    return model

class GACLoss(torch.nn.Module):
    def __init__(self, base, model_ref):
        super().__init__(); self.base = base; self.ref = model_ref
    def forward(self, preds, batch):
        d = self.base(preds, batch)
        fs, img = _S.pop('s', None), _S.pop('img', None)
        if fs is None or img is None: return d
        lab, ln = batch[-2], batch[-1]                          # ctc_label, ctc_length (2 key cuối của KeepKeys)
        ch = _S['chars']; H, Wd = img.shape[-2:]
        texts = [''.join(ch[int(c)] for c in lab[i, :int(ln[i])]) for i in range(lab.shape[0])]
        clean = torch.stack([render(t, H, Wd) for t in texts]).to(img.device)
        teacher = self.ref[0]._gac_teacher
        if next(teacher.parameters()).device != img.device: teacher.to(img.device)
        with torch.no_grad(): teacher.encoder(clean)
        ft = _S.pop('t')
        g = (1 - (_pool(fs) * _pool(ft).detach()).sum(1)).mean()
        d['loss'] = d['loss'] + ALPHA * g; d['gac_loss'] = g.detach()
        return d
'''
with open(f'{W}/gac.py', 'w', encoding='utf-8') as f: f.write(GAC_SRC)

TRAIN_GAC_SRC = r'''
import os, sys, runpy
REPO = os.environ['ART_REPO']; sys.path.insert(0, REPO); sys.path.insert(0, os.environ['ART_W']); os.chdir(REPO)
import gac, openrec.modeling as M, openrec.losses as L
from openrec.preprocess.ctc_label_encode import CTCLabelEncode
chars = CTCLabelEncode(25, os.environ['ART_DICT'], False).character      # ['blank', ...] đúng như lúc encode
_bm, _bl, REF = M.build_model, L.build_loss, []
def bm(cfg):
    m = gac.apply(_bm(cfg), cfg, _bm, os.environ['ART_B0'], chars); REF.append(m); return m
M.build_model = bm
L.build_loss = lambda cfg: gac.GACLoss(_bl(cfg), REF)
sys.argv = ['tools/train_rec.py'] + sys.argv[1:]
runpy.run_path('tools/train_rec.py', run_name='__main__')
'''
with open(f'{W}/train_gac.py', 'w', encoding='utf-8') as f: f.write(TRAIN_GAC_SRC)

def train_gac():
    out = f'{W}/runs/{NAME}'
    if os.path.exists(f'{out}/DONE'): print(f'[train] {NAME}: đã xong'); return out
    p, out = make_train_cfg(NAME, RECIPES[BASE])
    env = dict(os.environ, ART_REPO=REPO, ART_W=W, ART_B0=B0_CKPT, ART_DICT=DICT_PATH, GAC_ALPHA=str(GAC_ALPHA),
               GAC_STAGE=str(GAC_STAGE), GAC_COLS=str(GAC_COLS),
               PYTHONPATH=f'{W}/compat:{W}:{REPO}:' + os.environ.get('PYTHONPATH', ''))
    ngpu = max(1, torch.cuda.device_count())
    cmd = [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={ngpu}', '--master_port=29513',
           f'{W}/train_gac.py', '--c', p]
    print(f'[train] {NAME} | recipe={RECIPES[BASE]} | α={GAC_ALPHA}\n  log: {out}/train.log')
    t0 = time.time(); keep = re.compile(r'best metric|cur metric|global_step: \d*[05]00\b|Error|Traceback|GAC|resume')
    with open(f'{out}/train.log', 'w') as lf:
        pr = subprocess.Popen(cmd, env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in pr.stdout:
            lf.write(line)
            if keep.search(line): print('  ', line.rstrip()[:300])
        pr.wait()
    if pr.returncode != 0:
        print(open(f'{out}/train.log').read()[-4000:]); raise RuntimeError(f'train {NAME} lỗi (xem {out}/train.log)')
    open(f'{out}/DONE', 'w').write(f'{time.time() - t0:.0f}s'); print(f'[train] {NAME} xong sau {(time.time() - t0) / 3600:.2f} giờ')
    return out

register_tests()
predict('B0_goc', svtr_cfg(B0_CKPT), B0_CKPT)                        # baseline (bỏ qua nếu đã có)
if os.path.exists(f'{W}/runs/{BASE}/best.pth'): predict(BASE, svtr_cfg(B0_CKPT), f'{W}/runs/{BASE}/best.pth')
else: print(f'[cảnh báo] không thấy {W}/runs/{BASE}/best.pth (phiên mới?) -> chỉ so được với B0_goc')
out = train_gac()
predict(NAME, svtr_cfg(B0_CKPT), f'{out}/best.pth')
report(pairs=[('B0_goc', NAME), (BASE, NAME)])
os.chdir('/kaggle/working')
print(f'\n[ĐỌC KẾT QUẢ] Test chính = union_art; đối chứng = iiit5k_test. wordart_bench là DEV (chọn checkpoint) -> không phải test.\n'
      f'  GAC có đóng góp nếu "{NAME} - {BASE}" trên union_art có Δacc > 0 và CI95 không chứa 0.')
