from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT/'src'))

EXPECTED_CKPT='684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6'
EXPECTED_CFG='f1ab35a0f5e44d3c24c940cdf1a07404068ad9191bad9fe232ad4fcf38bbbe29'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):
            h.update(buf)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description='Frozen SCOPE-CTC V4 one-image inference, no labels required')
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--openocr-repo',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--output-json',type=Path)
    args=p.parse_args()
    for file in [args.image,args.checkpoint,args.config]:
        if not file.is_file():p.error(f'File not found: {file}')
    if not args.openocr_repo.is_dir():p.error(f'OpenOCR directory not found: {args.openocr_repo}')
    for path,expected in [(args.checkpoint,EXPECTED_CKPT),(args.config,EXPECTED_CFG)]:
        actual=sha256(path)
        if actual!=expected:p.error(f'Frozen asset SHA-256 mismatch: {path} expected {expected}, found {actual}')
    import torch
    from PIL import Image
    import longtext_final as LT
    import unified_router as UNI
    import scope_ctc_jit as JIT
    import scope_beam_jit_v4 as BEAM
    import scope_coverage_runtime as SCOPE
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable. Use --device cpu (slow) or install GPU PyTorch.')
    net,chars=LT.load_official_svtr(args.openocr_repo,args.config,args.checkpoint,args.device)
    recognizer=UNI.UnifiedRecognizer(net,chars,device=args.device,batch_size=args.batch_size)
    JIT.activate(strict=True)
    JIT.activate_forced(strict=True)
    BEAM.activate(strict=True)
    with Image.open(args.image) as im, torch.inference_mode():
        result=SCOPE.predict_image(im.convert('RGB'),recognizer)
    keep=('original_svtrv2_full12','safe_scope','safe_scope_active','coverage_decision','route',
          'safe_scope_total_ms','coverage_probe_ms','scope_decoder_ms','paired_scale_forward_ms')
    report={k:result.get(k) for k in keep}
    print(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf8')

if __name__=='__main__':main()
