import argparse


def str2bool(v):
    """argparse's `type=bool` maps any non-empty string to True, so `--flag False`
    silently enables the flag. Used for new boolean flags; the pre-existing ones
    (--resume, --multi_gpu, --tboard) still carry that behaviour."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean value, got {v!r}')


def get_parser_main_model():
    parser = argparse.ArgumentParser()
    # basic parameters training related
    parser.add_argument('--model_name', type=str, default='main_model', choices=['main_model', 'neural_raster'], help='current model_name')
    parser.add_argument("--language", type=str, default='eng', choices=['eng', 'chn'])
    parser.add_argument('--bottleneck_bits', type=int, default=512, help='latent code number of bottleneck bits')
    parser.add_argument('--char_num', type=int, default=52, help='number of glyphs, original is 52')
    parser.add_argument('--ref_nshot', type=int, default=4, help='reference number')    
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=8, help='batch size when do validation')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--max_seq_len', type=int, default=51, help='maximum length of sequence')
    parser.add_argument('--dim_seq', type=int, default=12, help='the dim of each stroke in a sequence, 4 + 8, 4 is cmd, and 8 is args')
    parser.add_argument('--dim_seq_short', type=int, default=9, help='the short dim of each stroke in a sequence, 1 + 8, 1 is cmd class num, and 8 is args')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden_size')
    parser.add_argument('--dim_seq_latent', type=int, default=512, help='sequence encoder latent dim')
    parser.add_argument('--ngf', type=int, default=16, help='the basic num of channel in image encoder and decoder')
    parser.add_argument('--n_aux_pts', type=int, default=6, help='the number of aux pts in bezier curves for additional supervison')
    # experiment related
    parser.add_argument('--seed', type=int, default=1111, help='random seed for torch, cuda, numpy and random')
    parser.add_argument('--random_index', type=str, default='00')
    parser.add_argument('--name_ckpt', type=str, default='600_192921.ckpt')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--resume', type=bool, default=False, help='resume training from experiments/<name_exp>/checkpoints/<name_ckpt>')
    parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs')
    parser.add_argument('--n_samples', type=int, default=20, help='the number of samples for each glyph when testing')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--ref_char_ids', type=str, default='0,1,26,27', help='default is A, B, a, b')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--name_exp', type=str, default='dvf')
    parser.add_argument('--data_root', type=str, default='./data/vecfont_dataset/')
    parser.add_argument('--freq_ckpt', type=int, default=50, help='save checkpoint frequency of epoch')
    parser.add_argument('--max_ckpt_keep', type=int, default=1, help='keep only the N checkpoints with the lowest validation loss (plus the latest, for resuming); -1 keeps all')
    parser.add_argument('--freq_sample', type=int, default=500, help='sample train output of steps')
    parser.add_argument('--freq_log', type=int, default=50, help='freq of showing logs')
    parser.add_argument('--freq_val', type=int, default=500, help='sample validate output of steps')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visulize loss')
    parser.add_argument('--wandb', type=str2bool, default=True, help='mirror scalar logging to Weights & Biases (metrics only, no artifacts); degrades to a no-op if wandb is not installed')

    # loss weight
    parser.add_argument('--kl_beta', type=float, default=0.01, help='latent code kl loss beta')
    parser.add_argument('--loss_w_pt_c', type=float, default=0.001 * 10, help='the weight of perceptual content loss')
    parser.add_argument('--loss_w_l1', type=float, default=1.0 * 10, help='the weight of image reconstruction l1 loss')
    parser.add_argument('--loss_w_cmd', type=float, default=1.0, help='the weight of cmd loss')
    parser.add_argument('--loss_w_args', type=float, default=1.0, help='the weight of args loss')
    parser.add_argument('--loss_w_aux', type=float, default=0.01, help='the weight of pts aux loss')
    parser.add_argument('--loss_w_smt', type=float, default=10., help='the weight of smooth loss')

    # Stage 2 experiment flags. Wired 2026-08-03. Every default reproduces the
    # pre-change behaviour exactly, so the 3-seed noise floor stays comparable to
    # everything launched afterwards. See PROJECT_PLAN.md 3.4.
    #   enc_noise_std_*   -> models/transformers.py, Transformer.forward   (E9)
    #   enc_final_norm    -> models/transformers.py, forward + att_residual (E1)
    #   dropout           -> both transformer stacks                        (E10)
    parser.add_argument('--enc_noise_std_train', type=float, default=1.0, help='[E9] sigma of the gaussian perturbation on the sequence-encoder output during training; 1.0 is the hardcoded original')
    parser.add_argument('--enc_noise_std_test', type=float, default=1.0, help='[E9] sigma of the same perturbation at val/test time; the only source of stochasticity across the n_samples candidates, so 0 makes all n_samples identical')
    parser.add_argument('--enc_final_norm', type=str2bool, default=False, help='[E1] add a terminal LayerNorm(512) to both sequence-encoder pre-norm stacks; the decoder already has one, the encoder does not')
    parser.add_argument('--dropout', type=float, default=0.0, help='[E10] dropout rate shared by the attention and feed-forward sublayers of both transformer stacks; every one of them is hardcoded to 0.0 upstream')

    # Tier 2 experiment flags. Wired 2026-08-04, same discipline as Tier 1: every
    # default reproduces the pre-change behaviour exactly, so the 3-seed noise floor
    # and the whole Tier 1 table stay comparable to everything launched afterwards.
    # See PROJECT_PLAN.md 3.5.
    #   args_label_smooth_sigma -> models/transformers.py, Transformer.loss      (E8)
    #   n_args_bins             -> arg_embed / args_fcn / reshapes / one_hot /
    #                              numericalize / denumericalize                 (E13)
    #   arg_embed_pad_idx       -> models/transformers.py, SVGEmbedding          (E13)
    #   n_layers_refine         -> models/transformers.py, Transformer_decoder   (E3)
    #   lr_schedule / lr_warmup_steps / lr_min_factor -> train.py                (E14)
    parser.add_argument('--args_label_smooth_sigma', type=float, default=0.0, help='[E8] std, in bins, of the discretized Gaussian replacing the one-hot argument target; 0 keeps the original permutation-invariant one-hot cross-entropy')
    parser.add_argument('--n_args_bins', type=int, default=128, help='[E13] number of quantization bins for the coordinate arguments; Sec. 3.1 of the paper specifies 256, the released code uses 128')
    parser.add_argument('--arg_embed_pad_idx', type=str2bool, default=True, help='[E13] keep padding_idx=0 on SVGEmbedding.arg_embed; bin 0 is a legitimate coordinate, and padding_idx freezes its embedding row at its init value')
    parser.add_argument('--n_layers_refine', type=int, default=1, help='[E3] depth of the parallel self-refinement decoder; Sec. 3.3 describes it as 2 layers, the released code clones 1')
    parser.add_argument('--lr_schedule', type=str, default='exp', choices=['exp', 'warmup_cosine'], help='[E14] exp is the original per-epoch ExponentialLR(gamma=0.997); warmup_cosine is per-step linear warmup then cosine decay to the epoch budget')
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help='[E14] linear warmup length in optimizer steps; only read when --lr_schedule warmup_cosine')
    parser.add_argument('--lr_min_factor', type=float, default=0.05, help='[E14] floor of the cosine decay as a fraction of --lr; only read when --lr_schedule warmup_cosine')

    # Tier 3 experiment flags. Wired 2026-08-04, same discipline again: every default
    # reproduces the released behaviour, so the Tier 1 and Tier 2 tables and the seed
    # floor all stay valid references. See PROJECT_PLAN.md 3.6.
    #   kl_beta          -> train.py loss sum                      (E12, pre-existing flag)
    #   ngf              -> image_encoder / image_decoder widths   (E4,  pre-existing flag)
    #   bottleneck_bits  -> modality_fusion + image_decoder        (E5,  pre-existing flag,
    #                       needs the z_proj added in this commit to be usable at all)
    #   optimizer        -> train.py                               (E11)
    #   img_norm         -> models/norms.py -> model_main.py       (E2)
    #   ema_decay        -> train.py                               (E15)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='[E11] adam is the original torch.optim.Adam, where --weight_decay is L2-in-the-gradient and interacts with the adaptive scale; adamw decouples it. AdamW is already imported in train.py and unused')
    parser.add_argument('--img_norm', type=str, default='layer', choices=['layer', 'group', 'batch', 'instance'], help="[E2] normalization in the image encoder and decoder. 'layer' is the original spatial LayerNorm([C,H,W]), which couples channel and spatial statistics and discards per-channel scale; the other three normalize per channel")
    parser.add_argument('--img_norm_groups', type=int, default=32, help='[E2] target group count for --img_norm group; halved automatically until it divides the channel count, so the ngf-wide first layer still constructs')
    parser.add_argument('--ema_decay', type=float, default=0.0, help='[E15] decay of an exponential moving average of the weights, evaluated and checkpointed in place of the raw weights. 0 disables it and keeps the original behaviour; 0.999 is the usual value')
    parser.add_argument('--ema_warmup_steps', type=int, default=0, help='[E15] steps before the EMA starts tracking; the shadow is initialized from the weights, so 0 is fine and this exists only for the record')

    return parser
