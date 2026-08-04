import os
import random
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from dataloader import get_loader
from models import util_funcs
from models.model_main import ModelMain
from options import get_parser_main_model
from data_utils.svg_utils import render
import checkpoint_log

try:
    import wandb
except ImportError:
    wandb = None

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_val_loss(model_main, val_loader, opts):
    loss_val = {'img':{'l1':0.0, 'vggpt':0.0}, 'svg':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0},
                'svg_para':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0}}
    with torch.no_grad():
        model_main.eval()
        for val_data in val_loader:
            for key in val_data: val_data[key] = val_data[key].cuda()
            ret_dict_val, loss_dict_val = model_main(val_data, mode='val')
            for loss_cat in ['img', 'svg', 'svg_para']:
                for key, _ in loss_val[loss_cat].items():
                    loss_val[loss_cat][key] += loss_dict_val[loss_cat][key]
        model_main.train()

    for loss_cat in ['img', 'svg', 'svg_para']:
        for key, _ in loss_val[loss_cat].items():
            loss_val[loss_cat][key] /= len(val_loader)

    # svg_para is the parallel/refinement decoder's loss. test_few_shot.py scores
    # sampled_svg_2, which comes from that decoder (not the teacher-forced `svg` one), so a
    # selection metric that omits it can prefer a checkpoint that isn't the one that will
    # actually score best at test time. Include both, matching the training objective.
    val_metric = float(opts.loss_w_l1 * loss_val['img']['l1'] + opts.loss_w_pt_c * loss_val['img']['vggpt']
                        + loss_val['svg']['total'] + loss_val['svg_para']['total'])
    return loss_val, val_metric

def prune_checkpoints(dir_ckpt, dir_log, latest_path, max_keep):
    """Keep the max_keep checkpoints with the lowest logged val_metric, plus latest_path."""
    ranked = sorted(checkpoint_log.read_all(dir_log), key=lambda r: float(r['val_metric']))
    keep = {latest_path}
    keep.update(os.path.join(dir_ckpt, r['checkpoint']) for r in ranked[:max_keep])

    for fname in os.listdir(dir_ckpt):
        if not fname.endswith('.ckpt'):
            continue
        path = os.path.join(dir_ckpt, fname)
        if path not in keep and os.path.exists(path):
            os.remove(path)

def train_main_model(opts):
    setup_seed(opts.seed)
    dir_exp = os.path.join("./experiments", opts.name_exp)
    dir_sample = os.path.join(dir_exp, "samples")
    dir_ckpt = os.path.join(dir_exp, "checkpoints")
    dir_log = os.path.join(dir_exp, "logs")
    logfile_train = open(os.path.join(dir_log, "train_loss_log.txt"), 'a' if opts.resume else 'w')
    logfile_val = open(os.path.join(dir_log, "val_loss_log.txt"), 'a' if opts.resume else 'w')

    train_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size, opts.mode)
    val_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size_val, 'test')

    model_main = ModelMain(opts)

    if torch.cuda.is_available() and opts.multi_gpu:
        model_main = torch.nn.DataParallel(model_main)
    
    model_main.cuda()

    parameters_all = [{"params": model_main.img_encoder.parameters()}, {"params": model_main.img_decoder.parameters()},
                        {"params": model_main.modality_fusion.parameters()}, {"params": model_main.transformer_main.parameters()},
                        {"params": model_main.transformer_seqdec.parameters()}]

    optimizer = Adam(parameters_all, lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)

    start_epoch = opts.init_epoch
    if opts.resume:
        ckpt_path = os.path.join(dir_ckpt, opts.name_ckpt)
        checkpoint = torch.load(ckpt_path, map_location='cuda')
        model_main.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        start_epoch = checkpoint['n_epoch'] + 1
        print(f"Resumed from {ckpt_path}, starting at epoch {start_epoch}")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

    if opts.tboard:
        writer = SummaryWriter(dir_log)

    use_wandb = bool(opts.wandb) and wandb is not None
    if opts.wandb and wandb is None:
        print("WARNING: --wandb is set but wandb is not installed; continuing without it. `pip install wandb`")
    if use_wandb:
        wandb.init(project="deepvecfont-v2", name=opts.name_exp, config=vars(opts), tags=[opts.language])

    loss_img_items = ['l1', 'vggpt']
    loss_svg_items = ['total', 'cmd', 'args', 'aux', 'smt']

    for epoch in range(start_epoch, opts.n_epochs):
        for idx, data in enumerate(train_loader):
            for key in data: data[key] = data[key].cuda()
            ret_dict, loss_dict = model_main(data)

            loss = opts.loss_w_l1 * loss_dict['img']['l1'] + opts.loss_w_pt_c * loss_dict['img']['vggpt'] + opts.kl_beta * loss_dict['kl'] \
                    + loss_dict['svg']['total'] + loss_dict['svg_para']['total']

            # perform optimization
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()
            batches_done = epoch * len(train_loader) + idx + 1 
            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.6f}, "
                f"img_l1_loss: {opts.loss_w_l1 * loss_dict['img']['l1'].item():.6f}, "
                f"img_pt_c_loss: {opts.loss_w_pt_c * loss_dict['img']['vggpt']:.6f}, "
                f"svg_total_loss: {loss_dict['svg']['total'].item():.6f}, "
                f"svg_cmd_loss: {opts.loss_w_cmd * loss_dict['svg']['cmd'].item():.6f}, "
                f"svg_args_loss: {opts.loss_w_args * loss_dict['svg']['args'].item():.6f}, "
                f"svg_smooth_loss: {opts.loss_w_smt * loss_dict['svg']['smt'].item():.6f}, "
                f"svg_aux_loss: {opts.loss_w_aux * loss_dict['svg']['aux'].item():.6f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                f"Step: {batches_done}"
            )
            if batches_done % opts.freq_log == 0:
                logfile_train.write(message + '\n')
                print(message)
                if opts.tboard:
                    writer.add_scalar('Loss/loss', loss.item(), batches_done)
                    for item in loss_img_items:
                        writer.add_scalar(f'Loss/img_{item}', loss_dict['img'][item].item(), batches_done)
                    for item in loss_svg_items:
                        writer.add_scalar(f'Loss/svg_{item}', loss_dict['svg'][item].item(), batches_done)
                    for item in loss_svg_items:
                        writer.add_scalar(f'Loss/svg_para_{item}', loss_dict['svg_para'][item].item(), batches_done)
                    writer.add_scalar('Loss/img_kl_loss', opts.kl_beta * loss_dict['kl'].item(), batches_done)
                    writer.add_image('Images/trg_img', ret_dict['img']['trg'][0], batches_done)
                    writer.add_image('Images/img_output', ret_dict['img']['out'][0], batches_done)

                if use_wandb:
                    # Same tag names as TensorboardX so the two dashboards read identically.
                    # One dict per step: repeated wandb.log calls at the same step can drop keys.
                    # Images are deliberately not mirrored (metrics only, no artifact uploads).
                    wandb_log = {'Loss/loss': loss.item()}
                    for item in loss_img_items:
                        wandb_log[f'Loss/img_{item}'] = loss_dict['img'][item].item()
                    for item in loss_svg_items:
                        wandb_log[f'Loss/svg_{item}'] = loss_dict['svg'][item].item()
                    for item in loss_svg_items:
                        wandb_log[f'Loss/svg_para_{item}'] = loss_dict['svg_para'][item].item()
                    wandb_log['Loss/img_kl_loss'] = opts.kl_beta * loss_dict['kl'].item()
                    wandb_log['lr'] = optimizer.param_groups[0]['lr']
                    wandb_log['epoch'] = epoch
                    wandb.log(wandb_log, step=batches_done)

            if opts.freq_sample > 0 and batches_done % opts.freq_sample == 0:
                
                img_sample = torch.cat((ret_dict['img']['trg'].data, ret_dict['img']['out'].data), -2)
                save_file = os.path.join(dir_sample, f"train_epoch_{epoch}_batch_{batches_done}.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)    
                
            if opts.freq_val > 0 and batches_done % opts.freq_val == 0:

                loss_val, last_val_metric = compute_val_loss(model_main, val_loader, opts)

                if opts.tboard:
                    for loss_cat in ['img', 'svg', 'svg_para']:
                        for key, _ in loss_val[loss_cat].items():
                            writer.add_scalar(f'VAL/loss_{loss_cat}_{key}', loss_val[loss_cat][key], batches_done)
                    writer.add_scalar('VAL/val_metric', last_val_metric, batches_done)

                if use_wandb:
                    wandb_val_log = {f'VAL/loss_{loss_cat}_{key}': float(loss_val[loss_cat][key])
                                     for loss_cat in ['img', 'svg', 'svg_para']
                                     for key in loss_val[loss_cat]}
                    wandb_val_log['VAL/val_metric'] = last_val_metric
                    wandb.log(wandb_val_log, step=batches_done)

                val_msg = (
                    f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                    f"Val loss img l1: {loss_val['img']['l1']: .6f}, "
                    f"Val loss img pt: {loss_val['img']['vggpt']: .6f}, "
                    f"Val loss total: {loss_val['svg']['total']: .6f}, "
                    f"Val loss cmd: {loss_val['svg']['cmd']: .6f}, "
                    f"Val loss args: {loss_val['svg']['args']: .6f}, "
                    f"Val loss para total: {loss_val['svg_para']['total']: .6f}, "
                    f"Val metric: {last_val_metric: .6f}, "
                )

                logfile_val.write(val_msg + "\n")
                print(val_msg)
        

        scheduler.step()

        if epoch % opts.freq_ckpt == 0:
            loss_val, last_val_metric = compute_val_loss(model_main, val_loader, opts)
            ckpt_name = f'{epoch}_{batches_done}.ckpt'
            ckpt_path = os.path.join(dir_ckpt, ckpt_name)
            if opts.multi_gpu:
                torch.save({'model':model_main.module.state_dict(), 'opt':optimizer.state_dict(), 'n_epoch':epoch, 'n_iter':batches_done}, ckpt_path)
            else:
                torch.save({'model':model_main.state_dict(), 'opt':optimizer.state_dict(), 'n_epoch':epoch, 'n_iter':batches_done}, ckpt_path)

            checkpoint_log.append(dir_log, epoch, batches_done, ckpt_name, last_val_metric, loss_val)

            if use_wandb:
                wandb.log({'CKPT/val_metric': last_val_metric, 'CKPT/epoch': epoch}, step=batches_done)

            if opts.max_ckpt_keep > 0:
                prune_checkpoints(dir_ckpt, dir_log, ckpt_path, opts.max_ckpt_keep)

    logfile_train.close()
    logfile_val.close()
    if use_wandb:
        wandb.finish()

def backup_code(name_exp):
    os.makedirs(os.path.join('experiments', name_exp, 'code'), exist_ok=True)
    shutil.copy('models/transformers.py', os.path.join('experiments', name_exp, 'code', 'transformers.py') )
    shutil.copy('models/model_main.py', os.path.join('experiments', name_exp, 'code', 'model_main.py'))
    shutil.copy('models/image_encoder.py', os.path.join('experiments', name_exp, 'code', 'image_encoder.py'))
    shutil.copy('models/image_decoder.py', os.path.join('experiments', name_exp, 'code', 'image_decoder.py'))
    shutil.copy('./train.py', os.path.join('experiments', name_exp, 'code', 'train.py'))
    shutil.copy('./options.py', os.path.join('experiments', name_exp, 'code', 'options.py'))

def train(opts):
    if opts.model_name == 'main_model':
        train_main_model(opts)
    elif opts.model_name == 'others':
        train_others(opts)
    else:
        raise NotImplementedError

def main():
    
    opts = get_parser_main_model().parse_args()
    opts.name_exp = opts.name_exp + '_' + opts.model_name
    os.makedirs("./experiments", exist_ok=True)
    debug = True
    # Create directories
    experiment_dir = os.path.join("./experiments", opts.name_exp)
    backup_code(opts.name_exp)
    os.makedirs(experiment_dir, exist_ok=debug)  # False to prevent multiple train run by mistake
    os.makedirs(os.path.join(experiment_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    print(f"Training on experiment {opts.name_exp}...")
    # Dump options
    with open(os.path.join(experiment_dir, "opts.txt"), "w") as f:
        for key, value in vars(opts).items():
            f.write(str(key) + ": " + str(value) + "\n")
    train(opts)

if __name__ == "__main__":
    main()
