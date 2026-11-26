"""Single-crop deployment smoke interface."""
import argparse, json
from PIL import Image
from .ultimate_pipeline import UltimateSTRSystem
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--image', required=True)
    a=p.parse_args()
    with Image.open(a.image) as im, UltimateSTRSystem.from_config(a.config) as system:
        result=system.predict(im.convert('RGB'))
        print(json.dumps({'text':result.text,'method':result.method,'latency_ms':result.latency_ms},ensure_ascii=False))
if __name__=='__main__':main()
