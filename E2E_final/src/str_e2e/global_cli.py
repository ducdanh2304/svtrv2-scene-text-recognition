"""CLI for localized text crops, with the frozen legacy modes still available."""
import argparse
import json
from PIL import Image
from .global_pipeline import GlobalSTRSystem, GLOBAL_MODES, MODES


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--mode',choices=sorted(GLOBAL_MODES|MODES),default='global_spatial')
    p.add_argument('--details',action='store_true')
    a=p.parse_args(argv)
    with Image.open(a.image) as im, GlobalSTRSystem.from_config(a.config) as s:
        pred=s.predict(im.convert('RGB'),mode=a.mode)
        print(json.dumps(pred.as_dict() if a.details else {'text':pred.text,'method':pred.method},ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
