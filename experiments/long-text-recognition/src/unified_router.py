"""A fully source-image/CTC-driven router; historical IDs/labels are never input."""
from __future__ import annotations
from PIL import Image
import time
import torch
import longtext_final as LT
import longtext_v3 as V3
import orientation_probe as OP
import density_probe as DP

class UnifiedRecognizer:
    def __init__(self,net,chars,device='cuda',batch_size=2):
        self.base=V3.EvidenceLongTextRecognizer(net,chars,device=device,batch_size=batch_size)
        self.orientation=OP.OrientationProbe(net,chars,device=device,batch_size=batch_size)
        self.dense=DP.Probe(net,chars,device=device,batch_size=batch_size)
        self.chars=list(chars)

    @staticmethod
    def classify(w,h,n):
        # Original raw-pixel geometry ONLY. ID, filename, GT and reference prediction unused.
        if int(w)<=int(h):
            assert int(n)==1, ('Tall expected a single original chunk',w,h,n)
            return 'orientation'
        if int(n)==1:
            return 'dense'
        return 'v3'

    def dense3_x3(self,images):
        # Reproduces density_probe.Probe.batch(...)["force_3_width_x3"] EXACTLY,
        # without paying for the five unused ablation branches.
        tensors=[];meta=[]
        for im in images:
            parts=DP.forced_crops(im,3)
            start=len(tensors)
            tensors.extend(DP.resized_tensor(p,3) for p,_,_ in parts)
            spans=[(a,b) for _,a,b in parts]
            meta.append((start,len(parts),spans))
        logits=self.dense._forward_tensors(tensors)
        ret=[]
        for start,n,spans in meta:
            cur=logits[start:start+n]
            if n==1:
                pred=LT.normalize_ltb(LT.greedy_ctc(cur[0],self.chars))
            else:
                events=[V3.ctc_events(x,self.chars,a,b) for x,(a,b) in zip(cur,spans)]
                pred=V3.stitch_evidence(events,spans,use_confidence=False,geometric_fallback=True)
            ret.append({'pred':pred,'dense_n_crops':n,'dense_ctc_Ts':[len(x) for x in cur]})
        return ret,{'dense_extra_tensors':len(tensors)}

    @torch.inference_mode()
    def infer_batch(self,images):
        images=[im.convert('RGB') for im in images]
        t0=time.perf_counter()
        # Independent matched baseline on EVERY image; the same official v3 Geometry.
        baseline=self.base.predict_many(images,method='v3_geom',return_info=True)
        routings=[self.classify(im.width,im.height,b['num_windows']) for im,b in zip(images,baseline)]
        rows=[{'route':route,'baseline_pred':b['prediction'],'baseline_num_windows':b['num_windows'],
               'pred':b['prediction'],'angle_choice':'', 'dense_n_crops':0,
               'orientation_agreement_delta':None,'orientation_conf_delta':None}
              for route,b in zip(routings,baseline)]
        ori_idx=[i for i,r in enumerate(routings) if r=='orientation']
        if ori_idx:
            ori,ori_perf=self.orientation.infer_batch([images[i] for i in ori_idx])
            for i,v in zip(ori_idx,ori):
                # Assert two independent original-orientation implementations agree.
                assert v['original']==rows[i]['baseline_pred'], ('Tall original parity',i)
                rows[i]['pred']=v['select_overlap']
                rows[i]['angle_choice']=v['overlap_choice']
                rows[i]['orientation_agreement_delta']=float(v['cw_agreement']-v['ccw_agreement'])
                rows[i]['orientation_conf_delta']=float(v['cw_confidence']-v['ccw_confidence'])
        else:ori_perf={'num_tensors':0}
        dense_idx=[i for i,r in enumerate(routings) if r=='dense']
        if dense_idx:
            preds,dense_perf=self.dense3_x3([images[i] for i in dense_idx])
            for i,v in zip(dense_idx,preds):
                rows[i]['pred']=v['pred']
                rows[i]['dense_n_crops']=int(v['dense_n_crops'])
                rows[i]['dense_ctc_Ts']=v['dense_ctc_Ts']
        else:dense_perf={'dense_extra_tensors':0}
        for r in rows:
            r.setdefault('dense_ctc_Ts',[])
        return rows,{'elapsed_seconds':time.perf_counter()-t0,
                     'baseline_images':len(images),'orientation_images':len(ori_idx),'dense_images':len(dense_idx),
                     'orientation_extra_tensors':ori_perf['num_tensors'],
                     'dense_extra_tensors':dense_perf['dense_extra_tensors']}

def run_cpu_tests():
    class Dummy(torch.nn.Module):
        def forward(self,x):
            b,_,h,w=x.shape
            t=w//4
            logits=torch.zeros((b,t,3),device=x.device)
            logits[:,:,0]=2.0
            logits[:,max(1,t//4),1]=4.0
            return logits
    net=Dummy()
    p=UnifiedRecognizer(net,['blank','a','b'],device='cpu',batch_size=2)
    imgs=[Image.new('RGB',(18,210),'white'),Image.new('RGB',(120,30),'white'),Image.new('RGB',(330,20),'white')]
    pred,perf=p.infer_batch(imgs)
    assert [x['route'] for x in pred]==['orientation','dense','v3'], pred
    assert pred[0]['angle_choice'] in ('cw','ccw')
    assert pred[1]['dense_n_crops']==3
    assert pred[2]['baseline_num_windows']>=2
    assert all(isinstance(x['pred'],str) for x in pred)
    # Dense-only wrapper must decode IDENTICALLY to the pre-existing full probe.
    all_dense,_=p.dense.batch([imgs[1]])
    only,_=p.dense3_x3([imgs[1]])
    assert only[0]['pred']==all_dense['force_3_width_x3'][0]['prediction']
    assert only[0]['dense_ctc_Ts']==all_dense['force_3_width_x3'][0]['ctc_Ts']
    return {'passed':True,'routes':[x['route'] for x in pred],'dense_parity':True}
