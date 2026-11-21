import argparse,sys,torch,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'OpenOCR'))
from tools.engine.config import Config
from tools.engine.trainer import Trainer

class S2Trainer(Trainer):
 def _init_rec_model(self):
  super()._init_rec_model()
  m=self.model
  assert len(m.encoder.stages)==4 and not tuple(m.encoder.stages[3].parameters())
  m.requires_grad_(False)
  m.encoder.stages[1].requires_grad_(True)
  m.encoder.stages[2].requires_grad_(True)
  m.decoder.requires_grad_(True)
  assert not any(p.requires_grad for p in m.encoder.pope.parameters())
  assert not any(p.requires_grad for p in m.encoder.stages[0].parameters())
  assert all(p.requires_grad for p in m.encoder.stages[1].parameters())
  assert all(p.requires_grad for p in m.encoder.stages[2].parameters())
  assert all(p.requires_grad for p in m.decoder.parameters())
  # Trainer.train()/eval_step() toggles mode; keep BN statistics fixed as in prior S2.
  def keep_bn(module,inputs):
   for sub in module.modules():
    if isinstance(sub,torch.nn.modules.batchnorm._BatchNorm):sub.eval()
  m.register_forward_pre_hook(keep_bn)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 cfg=Config(a.config)
 out=Path(cfg.cfg['Global']['output_dir']);latest=out/'latest.pth'
 if latest.exists():
  saved=torch.load(latest,map_location='cpu',weights_only=False)
  if saved['epoch']>=cfg.cfg['Global']['epoch_num']:
   print('COMPLETE',latest,'epoch',saved['epoch']);return
  cfg.cfg['Global']['checkpoints']=str(latest)
 t=S2Trainer(cfg,mode='train_eval',task='rec');m=t.model
 if cfg.cfg['Global']['checkpoints'] is None:
  expected=torch.load(cfg.cfg['Global']['pretrained_model'],map_location='cpu',weights_only=False)['state_dict']
  actual=m.state_dict();assert set(actual)==set(expected)
  assert all(torch.equal(actual[k].detach().cpu(),v) for k,v in expected.items()),'Initial weights differ from converted official checkpoint'
 names=[n for n,p in m.named_parameters() if p.requires_grad]
 print('S2 TRAINABLE',len(names),'parameters',sum(p.numel() for p in m.parameters() if p.requires_grad))
 if a.smoke:
  b=next(iter(t.train_dataloader));tensor=[x.to(t.device) for x in b]
  m.train();pred=m(tensor[0],data=tensor[1:]);loss=t.loss_class(pred,tensor)['loss']
  assert bool(torch.isfinite(loss));loss.backward()
  assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.encoder.stages[1].parameters())
  assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.encoder.stages[2].parameters())
  assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.decoder.parameters())
  assert all(p.grad is None for p in m.encoder.stages[0].parameters())
  (out/'smoke.json').write_text(json.dumps({'trainable_params':sum(p.numel() for p in m.parameters() if p.requires_grad),
   'batch_shape':list(tensor[0].shape),'loss':float(loss.detach()),'stage2_gradient':True,
   'stage3_gradient':True,'decoder_gradient':True,'stage1_gradient':False},indent=2))
  print('SMOKE PASS',tuple(tensor[0].shape),'loss',float(loss.detach()))
  return
 t.train()
 assert latest.exists() and torch.load(latest,map_location='cpu',weights_only=False)['epoch']==cfg.cfg['Global']['epoch_num']
 print('COMPLETE',latest)
if __name__=='__main__':main()
