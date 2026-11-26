"""Create a separate runtime; reuse the original Global resource verifier unchanged."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def install_runtime(release, root):
    manifest=json.loads((release/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items():
        if sha(release/name)!=h: raise RuntimeError('Release hash mismatch: '+name)
    backend=root/'src/str_e2e/backends'
    for name,h in manifest['original_backends'].items():
        if sha(backend/name)!=h: raise RuntimeError('Global runtime differs: '+name)
    if sha(root/'global_release/global_calibration.locked.json')!=manifest['calibration_sha256']:
        raise RuntimeError('Calibration differs')
    for name,target in manifest['backend_map'].items():
        shutil.copyfile(release/'source'/name,backend/target)
        if sha(backend/target)!=manifest['files']['source/'+name]:raise RuntimeError('Copy mismatch: '+target)
    for name in ('ultimate_pipeline.py','ultimate_cli.py'):
        shutil.copyfile(release/name,root/'src/str_e2e'/name)
    dest=root/'ultimate_release'; dest.mkdir(exist_ok=False)
    shutil.copyfile(release/'manifest.json',dest/'manifest.json')
    print('ULTIMATE_RUNTIME_READY',root)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--work',required=True,type=Path);a=p.parse_args()
    release=Path(__file__).resolve().parent
    work=a.work.resolve(); checkout=release.parent.parent.resolve()
    if work==checkout or checkout in work.parents:raise RuntimeError('Use a fresh workspace outside the repository')
    if work.exists():raise FileExistsError(work)
    subprocess.run([sys.executable,str(release.parent/'scripts/prepare_global_runtime.py'),'--bundle',str(a.bundle.resolve(strict=True)),'--work',str(work)],check=True)
    install_runtime(release,work/'E2E_final')
if __name__=='__main__':main()
