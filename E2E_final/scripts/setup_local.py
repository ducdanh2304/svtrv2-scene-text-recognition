from pathlib import Path
root=Path(__file__).resolve().parents[1]
for a,b in [('configs/system.example.yaml','configs/system.local.yaml'),('configs/truc_fuse/r3_runtime.yml','configs/truc_fuse/r3_runtime.local.yml'),('configs/truc_fuse/waterec_runtime.yml','configs/truc_fuse/waterec_runtime.local.yml'),('checkpoints/official/config.yml','checkpoints/official/config.local.yml')]:
    dst=root/b
    if dst.exists(): print('KEEP',b); continue
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text((root/a).read_text().replace('__E2E_FINAL_ROOT__',str(root)))
    print('CREATED',b)
