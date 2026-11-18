from pathlib import Path
import ast, hashlib, json, sys

here = Path(__file__).resolve().parents[1]
notebook = json.loads((here/'notebooks/SAGE_STR_v4_Kaggle_RunAll.ipynb').read_text(encoding='utf-8'))
found = {}
for cell in notebook['cells']:
    s = ''.join(cell.get('source',[]))
    if s.startswith('%%writefile '):
        first, data = s.split('\n',1)
        found[first.replace('%%writefile ','',1).strip()] = data.encode('utf-8')
    else:
        try: tree=ast.parse(s)
        except SyntaxError: continue
        for call in ast.walk(tree):
            if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and
                call.func.attr=='write_text' and isinstance(call.func.value,ast.Call) and
                isinstance(call.func.value.func,ast.Name) and call.func.value.func.id=='Path'):
                try:
                    file=ast.literal_eval(call.func.value.args[0]); content=ast.literal_eval(call.args[0])
                except (ValueError,IndexError,TypeError): continue
                if file in ('sage_str_v2/ctc_evidence.py','sage_gpu_adapter.py'):
                    found[file]=content.encode('utf-8')

needed=['longtext_final.py','longtext_v3.py','orientation_probe.py','density_probe.py',
        'unified_router.py','sage_str_v2/selector.py','sage_str_v2/ctc_evidence.py',
        'sage_str_v3/selector.py','sage_str_v4/selector.py','sage_str_v4/batch.py','sage_gpu_adapter.py']
for name in needed:
    assert name in found, f'Not extracted from notebook: {name}'
    disk=(here/'src'/name).read_bytes()
    assert disk==found[name], f'Notebook-source mismatch: {name}'
manifest=json.loads((here.parents[2]/'reports/long_text/sage_str_v4/results/run_manifest.json').read_text(encoding='utf-8'))
sha=hashlib.sha256(found['sage_str_v4/selector.py']).hexdigest()
assert sha==manifest['v4_code_sha256'], f'Uploaded GPU selector SHA differs: {sha}'
print(f'SOURCE PARITY PASS: {len(needed)} module files match notebook; GPU selector sha256 {sha}')
