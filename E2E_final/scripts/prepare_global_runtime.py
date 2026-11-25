"""Restore an isolated byte-verified runtime from the original E2E resource bundle.
Resource restoration functions are taken from the full benchmark notebook.
Run with Python 3.12. Never overwrites a preexisting runtime.
"""
import argparse, collections, hashlib, json, re, shutil, subprocess, sys, tarfile, zipfile
from pathlib import Path, PurePosixPath
MANIFEST_SHA = 'cc2cd18aa63a8ecb492238fb7f76f013fff8437168068e6c96685b83a558782a'
EXPANDED = {}
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def safe_name(s):
    p=PurePosixPath(s)
    if p.is_absolute() or '..' in p.parts or chr(92) in s: raise ValueError('Unsafe path: '+s)
    return p

def resolve_directory_bundle(bundle, expected, search_root):
    # Resolve files moved by the input layout, never accept an unverified substitute.
    resolved = {}
    unresolved = {}
    for rel, h in expected.items():
        path = bundle / rel
        if path.is_file():
            resolved[rel] = path
        else:
            unresolved[rel] = h
    if unresolved:
        wanted = {Path(rel).name for rel in unresolved}
        inventory = collections.defaultdict(list)
        for path in search_root.rglob('*'):
            if path.name in wanted and path.is_file():
                inventory[path.name].append(path)
        for rel, h in unresolved.items():
            for path in sorted(inventory[Path(rel).name]):
                if sha(path) == h:
                    resolved[rel] = path
                    print('RELOCATED SHA PASS', rel, '->', path, flush=True)
                    break
    missing = sorted(set(expected) - set(resolved))
    return resolved, missing

def asset_tree(rel, label):
    if rel in EXPANDED:
        return EXPANDED[rel]
    temp = WORK / ('extract_' + label)
    unpack_tar(REPRO / rel, temp)
    return temp

def unpack_resources():
    if ZIP_PATH.is_dir():
        manifest=(ZIP_PATH/'SHA256SUMS.txt').read_bytes()
        assert hashlib.sha256(manifest).hexdigest()==MANIFEST_SHA,'SHA manifest differs from V1 release'
        expected={}
        for line in manifest.decode().splitlines():
            h,rel=line.split(maxsplit=1); rel=rel.lstrip('*'); safe_name(rel)
            assert re.fullmatch('[0-9a-f]{64}',h) and rel not in expected
            expected[rel]=h
        assert len(expected)==34
        sources, missing = resolve_directory_bundle(ZIP_PATH, expected, INPUT)
        archive_names = {
            'datasets/'+name+'.tar.gz' for name in
            ('iiit5k','u14m_artistic','blur_f008','blur_f018','blur_f030','ltb_clean_eval')
        } | {'models/e2e_v1_models.tar.gz', 'external/OpenOCR_source.tar.gz',
             'external/OpenOCR-WATERec_source.tar.gz'}
        EXPANDED.clear()
        for rel in missing:
            if rel not in archive_names:
                continue
            candidate = ZIP_PATH / rel[:-7]
            if candidate.is_dir() and any(p.is_file() for p in candidate.rglob('*')):
                EXPANDED[rel] = candidate
        missing = sorted(set(expected) - set(sources) - set(EXPANDED))
        report = {'bundle':str(ZIP_PATH), 'expected':len(expected),
                  'archive_files_present':len(sources), 'expanded_archives':{k:str(v) for k,v in EXPANDED.items()},
                  'missing':missing, 'resolved':{k:str(v) for k,v in sources.items()},
                  'archive_sha_status':'EXPANDED_ARCHIVE_BYTES_UNVERIFIABLE' if EXPANDED else 'EXACT_ARCHIVES_REQUIRED'}
        dump(OUT/'resource_inventory.json', report)
        print('RESOURCE INVENTORY:', len(sources), 'files +', len(EXPANDED), 'expanded directories /', len(expected), flush=True)
        if missing:
            raise FileNotFoundError('Missing resource or expanded directory: '+repr(missing))
        needed = sum(p.stat().st_size for p in sources.values())*3
        needed += sum(p.stat().st_size for root in EXPANDED.values() for p in root.rglob('*') if p.is_file())
        assert shutil.disk_usage(WORK).free > needed+2_000_000_000, 'Insufficient space for extraction and dependencies'
        for rel,source in sources.items():
            dst=REPRO/rel; dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dst)
            assert sha(dst)==expected[rel], 'Resource hash mismatch: '+rel
            print('SHA PASS',rel,flush=True)
        for rel,source in EXPANDED.items():
            print('EXPANDED INPUT (original archive SHA unavailable)',rel,'->',source,flush=True)
        (OUT/'SHA256SUMS.txt').write_bytes(manifest)
        return expected
    if not ZIP_PATH.is_file(): raise FileNotFoundError(f'Attach the complete V1 resource bundle via Add Input or edit RESOURCE_PATH in cell 1.')
    if not zipfile.is_zipfile(ZIP_PATH): raise ValueError('ZIP missing central directory / truncated. Use original full file, not chat attachment.')
    with zipfile.ZipFile(ZIP_PATH) as z:
        paths=[i.filename for i in z.infolist() if not i.is_dir()]
        for i in z.infolist(): safe_name(i.filename)
        if len(paths)!=len(set(paths)): raise ValueError('Duplicate ZIP members')
        found=[p for p in paths if PurePosixPath(p).name=='SHA256SUMS.txt']
        if len(found)!=1: raise ValueError('Expected exactly one SHA256SUMS.txt')
        manifest=z.read(found[0]); assert hashlib.sha256(manifest).hexdigest()==MANIFEST_SHA,'SHA manifest differs from the supplied V1 release'
        prefix=found[0][:-len('SHA256SUMS.txt')]
        expected={}
        for line in manifest.decode().splitlines():
            h,rel=line.split(maxsplit=1); rel=rel.lstrip('*'); safe_name(rel)
            assert re.fullmatch('[0-9a-f]{64}',h) and rel not in expected
            expected[rel]=h
        assert len(expected)==34, f'Unexpected number of manifest entries: {len(expected)}'
        free=shutil.disk_usage(WORK).free
        assert free>sum(i.file_size for i in z.infolist())*3+2_000_000_000,'Insufficient space for extraction and dependencies'
        for rel,h in expected.items():
            dst=REPRO/rel; dst.parent.mkdir(parents=True,exist_ok=True)
            with z.open(prefix+rel) as src,dst.open('wb') as f: shutil.copyfileobj(src,f,1<<20)
            assert sha(dst)==h,'Resource hash mismatch: '+rel
            print('SHA PASS',rel,flush=True)
        (OUT/'SHA256SUMS.txt').write_bytes(manifest)
    return expected

def unpack_tar(src,dst):
    dst.mkdir(parents=True,exist_ok=True)
    with tarfile.open(src,'r:gz') as t:
        for m in t.getmembers():
            safe_name(m.name)
            if not (m.isfile() or m.isdir() or m.issym() or m.islnk()): raise ValueError('Unsafe tar member: '+m.name)
        t.extractall(dst,filter='data')

def setup_assets():
    db_paths={'iiit5k':'datasets/iiit5k/evaluation/IIIT5k','u14m_artistic':'datasets/union_artistic/u14m/artistic','blur_f008':'datasets/router_motion_dev/blur_f008','blur_f018':'datasets/router_motion_dev/blur_f018','blur_f030':'datasets/router_motion_dev/blur_f030','ltb_clean_eval':'datasets/ltb_unpacked/ltb/long_lmdb'}
    for name,rel in db_paths.items():
        temp=asset_tree(f'datasets/{name}.tar.gz',name)
        candidates=list(temp.rglob('data.mdb'))
        assert len(candidates)==1,(name,candidates)
        dest=ROOT/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists(); dest.mkdir()
        shutil.copy2(candidates[0],dest/'data.mdb')
    model_temp=asset_tree('models/e2e_v1_models.tar.gz','models')
    model_manifest=json.loads((ROOT/'release_manifest.json').read_text())
    required={a['path'] for group in model_manifest['model_assets_not_uploaded'].values() for a in group}
    required.add('checkpoints/official/config.yml')
    for rel in sorted(required):
        candidates=[p for p in model_temp.rglob(Path(rel).name) if p.is_file() and p.relative_to(model_temp).as_posix().endswith(rel)]
        assert len(candidates)==1,(rel,candidates)
        dest=ROOT/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(candidates[0],dest)
    for name in ('OpenOCR','OpenOCR-WATERec'):
        temp=asset_tree(f'external/{name}_source.tar.gz',name)
        candidates=[p.parent for p in temp.rglob('openrec') if p.is_dir() and (p/'modeling/__init__.py').is_file() and (p.parent/'tools').is_dir()]
        assert len(candidates)==1,(name,candidates)
        dest=ROOT/'external'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists(); shutil.copytree(candidates[0],dest,symlinks=False)
    mapping=ROOT/'outputs/unified_v4/provenance_official/official_rgb_mapping.csv'
    mapping.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(REPRO/'metadata/official_rgb_mapping.csv',mapping)
    m=json.loads((ROOT/'release_manifest.json').read_text())
    for rel,h in m['files_sha256'].items(): assert sha(ROOT/rel)==h, 'Release file mismatch: '+rel
    for entries in m['model_assets_not_uploaded'].values():
        for a in entries:
            p=ROOT/a['path']; assert p.stat().st_size==a['size'] and sha(p)==a['sha256'],str(p)
    s=ROOT/'research_upstream/experiments/long-text-recognition/src'
    hashes=json.loads((s/'SOURCE_SHA256.json').read_text()); assert len(hashes)==17
    for name,h in hashes.items(): assert sha(s/name)==h,name
    for name in ('OpenOCR','OpenOCR-WATERec'):
        p=ROOT/'external'/name
        if (p/'.git').exists():
            rev=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
            if m['external_commits'][name] is not None: assert rev==m['external_commits'][name],(name,rev)
    external_audit={}
    for name in ('OpenOCR','OpenOCR-WATERec'):
        p=ROOT/'external'/name
        file_hashes={f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*'))
                     if f.is_file() and '.git' not in f.relative_to(p).parts}
        external_audit[name]={'expected_commit':m['external_commits'][name],
            'archive_sha_verified':f'external/{name}_source.tar.gz' not in EXPANDED,
            'git_metadata_present':(p/'.git').exists(), 'observed_file_sha256':file_hashes}
    dump(OUT/'external_input_audit.json',external_audit)
    # Preserve official config byte-for-byte; only machine paths in other local configs change.
    for a,b in [('configs/system.example.yaml','configs/system.local.yaml'),('configs/truc_fuse/r3_runtime.yml','configs/truc_fuse/r3_runtime.local.yml'),('configs/truc_fuse/waterec_runtime.yml','configs/truc_fuse/waterec_runtime.local.yml')]:
        (ROOT/b).write_bytes((ROOT/a).read_bytes().replace(b'__E2E_FINAL_ROOT__',str(ROOT).encode()))
    shutil.copyfile(ROOT/'checkpoints/official/config.yml',ROOT/'checkpoints/official/config.local.yml')
    assert sha(ROOT/'checkpoints/official/config.local.yml')==m['files_sha256']['checkpoints/official/config.yml']
    for rel in ['external_commits.txt','pip_freeze.txt','python_version.txt','nvidia_smi.txt']:
        shutil.copy2(REPRO/'metadata'/rel,OUT/('reference_'+rel))
    for rel in ['configs/system.local.yaml','configs/truc_fuse/r3_runtime.local.yml','configs/truc_fuse/waterec_runtime.local.yml']:
        shutil.copy2(ROOT/rel,OUT/Path(rel).name)
    print('RELEASE_FILES_PASS',len(m['files_sha256']),'SCOPE_SOURCE_PASS',len(hashes))

def main():
    global ZIP_PATH, INPUT, WORK, ROOT, OUT, REPRO
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True,help='New directory, outside tracked source')
    parser.add_argument('--eval-data',type=Path,help='Private evaluation files directory (optional for inference)')
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]
    rel=source/'global_release'
    ZIP_PATH=args.bundle.resolve(strict=True)
    INPUT=ZIP_PATH if ZIP_PATH.is_dir() else ZIP_PATH.parent
    WORK=args.work.resolve(); WORK.mkdir(parents=True,exist_ok=False)
    ROOT=WORK/'E2E_final'; ROOT.mkdir(); OUT=WORK/'setup_audit'; OUT.mkdir(); REPRO=WORK/'resources'
    archive=rel/'frozen_runtime.zip'
    if sha(archive)!='f4b3bdf27a6622de0dd94ba7df7fbebfbd69c5dc55938e0b6132e579bb115d90':
        raise RuntimeError('Frozen source archive hash mismatch')
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist(): safe_name(info.filename)
        z.extractall(ROOT)
    unpack_resources(); setup_assets()
    # Only add new release files AFTER verifying the original manifest.
    manifest=json.loads((rel/'overlay_manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        safe_name(name); p=source/name
        if sha(p)!=digest: raise RuntimeError('Overlay source hash mismatch: '+name)
        if name.startswith(('src/','tests/','global_release/')) and name!='global_release/frozen_runtime.zip' and not name.endswith('.ipynb'):
            dst=ROOT/name
            if dst.exists(): raise RuntimeError('Unexpected overwrite: '+name)
            dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dst)
    shutil.copyfile(rel/'overlay_manifest.json',ROOT/'global_release/overlay_manifest.json')
    if args.eval_data:
        private=args.eval_data.resolve(strict=True)
        hashes=json.loads((rel/'eval_manifest.json').read_text())
        for name in ('calibration_manifest.json','global_reference940.json'):
            src=private/name
            if sha(src)!=hashes[name]: raise RuntimeError('Evaluation resource hash mismatch: '+name)
            shutil.copyfile(src,ROOT/'global_release'/name)
        for name in ('global_joint.py','global_probability.py'):
            shutil.copyfile(ROOT/'src/str_e2e/backends'/name,ROOT/'global_release'/name)
    print('Runtime ready:',ROOT)
    print('Audit:',OUT)
    print('Set PYTHONPATH to',ROOT/'src')
if __name__=='__main__': main()
