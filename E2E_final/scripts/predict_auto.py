import argparse, json
from pathlib import Path
import yaml
from PIL import Image
from str_e2e.system import STRSystem
p=argparse.ArgumentParser();p.add_argument('image');args=p.parse_args()
root=Path(__file__).resolve().parents[1]
config=yaml.safe_load((root/'configs/system.local.yaml').read_text())
system=STRSystem(config)
try:
    with Image.open(args.image) as image:
        base,final=system.predict_pair(image.convert('RGB'),mode='auto')
    print(json.dumps({'route':final.route,'baseline':base.text,'prediction':final.text,'method':final.method},ensure_ascii=False))
finally:system.close()
