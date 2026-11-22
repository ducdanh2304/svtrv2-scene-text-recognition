# Tạo lại config TRAIN của R3_lowlr từ svtrv2_eval_config.yml (file này có trong artifact fuse-release).
# Bản config.yml gốc của run R3 không được lưu vào artifact; script này áp ĐÚNG các ghi đè mà
# make_train_cfg() trong kaggle_fuse_release_cell.py đã dùng, nên kết quả giống bản gốc (trừ đường dẫn).
# Dùng: python make_r3_train_config.py svtrv2_eval_config.yml r3_lowlr_train_config.yml
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
W = '/kaggle/working/art_pipeline'; OUT = f'{W}/runs/R3_lowlr'
B0 = '/kaggle/input/notebooks/trctrnthanh386/xulianh-deblur/restormer_truc/official_svtrv2_base/best.pth'
cfg = yaml.safe_load(open(src, encoding='utf-8'))
cfg['Global'].update(epoch_num=8, output_dir=OUT, pretrained_model=B0, checkpoints=None,
                     eval_batch_step=[0, 500], eval_epoch_step=[0, 1], save_epoch_step=[9, 1],
                     save_res_path=f'{OUT}/predicts.txt', use_amp=True, print_batch_step=50)
cfg['Optimizer']['lr'] = 1e-5
cfg['LRScheduler'] = dict(name='OneCycleLR', warmup_epoch=0.5, cycle_momentum=False)
for mode, p in [('Train', f'{W}/data/T20k_R50k/train'), ('Eval', f'{W}/data/wordart_dev')]:
    cfg[mode]['dataset']['data_dir_list'] = [p]
    cfg[mode]['sampler']['first_bs'] = 64
    cfg[mode]['loader']['batch_size_per_card'] = 64
    cfg[mode]['loader']['num_workers'] = 2
yaml.safe_dump(cfg, open(dst, 'w', encoding='utf-8'), allow_unicode=True, sort_keys=False)
print('->', dst)
