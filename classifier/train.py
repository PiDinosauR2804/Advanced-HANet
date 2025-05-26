from classifier.model import BertED
from classifier.exemplars import Exemplars
from utils.dataloader import (
    collect_dataset, collect_exemplar_dataset, 
    collect_sldataset, collect_from_json, 
    MAVEN_Dataset,
    DescriptionDataset, collect_eval_sldataset
)
from utils.computeLoss import compute_CLLoss, CrossEntropyLossWithWeight
from utils.tools import contrastive_loss_des, find_negative_labels, collate_description, balance_zero_with_nonzero
from utils.calcs import Calculator

import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time, sys, json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from copy import deepcopy
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR, LambdaLR, CyclicLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizerFast
import wandb
from loguru import logger
from tqdm.auto import tqdm
import optuna


def train(local_rank, args, trial=None):    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Configure logging
    os.makedirs(args.logs_dir, exist_ok=True)
    # --- Xoá handler mặc định ---
    logger.remove()
    # --- Thêm handler ghi log ra file ---
    date_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger.add(os.path.join(args.logs_dir, f"{date_str}.log"), rotation="1 MB", retention="10 days", enqueue=True, level="INFO")
    # --- Thêm handler ghi log qua tqdm.write ---
    # Ghi log ra console qua tqdm.write + có màu
    logger.level("CRITICAL", color="<bg red><white>")
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level="DEBUG",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file: >18}: {line: <4}</cyan> - <level>{message}</level>",
    )
    
    # set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # set device, whether to use cuda or cpu
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Thêm timestamp
    args.run_name = f"{args.dataset}_{args.task_num}_{args.shot_num}_{args.class_num}_{args.distill}_{args.cl_aug}_{args.joint_da_loss}_{args.skip_first_cl}_{timestamp}"

    # get streams from json file and permute them in pre-defined order
    # PERM = PERM_5 if args.task_num == 5 else PERM_10
    
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project_name,
            name = args.run_name,

            # track hyperparameters and run metadata
            config=args.__dict__,
            reinit=True
        )
    else:
        wandb.init(
            project=args.project_name,
            name = args.run_name,
            mode="disabled"
        )
    
    # Đọc dữ liệu
    streams, _ = collect_from_json(args.dataset, args.stream_root, 'stream', args)
    # streams = [streams[l] for l in PERM[int(args.perm_id)]] # permute the stream
    label2idx = {0:0}
    idx2label = {}
    
    for st in streams:
        for lb in st:
            if lb not in label2idx:
                label2idx[lb] = len(label2idx)
    
    for key, value in label2idx.items():
        idx2label[value] = key
    
    streams_indexed = [[label2idx[l] for l in st] for st in streams]
    
    # streams_indexed có dạng [[4, 5, 9, 11], [2, 1, 8, 33], ...] thể hiện thứ tự label class được học
    if args.backbone_path != "":
        model = BertED(args, args.backbone_path) # define model
    else:
        model = BertED(args) # define model
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay, eps=args.adamw_eps, betas=(0.9, 0.999)) #TODO: Hyper parameters
    
    def _lr_lambda_wapper(epoch):
        if epoch < args.epochs:
            stage = 0
            stage_ep = epoch
            stage_total_ep = args.epochs
        else:
            stage = (epoch - args.epochs) // int(args.epochs * args.task_ep_time) + 1
            stage_ep = epoch - args.epochs - (stage - 1) * int(args.epochs * args.task_ep_time)
            stage_total_ep = int(args.epochs * args.task_ep_time)
            
        def _lr_lambda(stage_ep, stage_total_ep):
            if stage_ep < args.warmup_ep:
                return max(stage_ep / args.warmup_ep, args.min_lr_ratio)
            else:
                return max(1.0 - ((stage_ep - args.warmup_ep) / (stage_total_ep - args.warmup_ep)), args.min_lr_ratio)

        return _lr_lambda(stage_ep, stage_total_ep)
    
    class LambdaLRFunc:
        def __init__(self, min_lr_ratio=0.1, warmup_ratio=0.1, stage_lr_ratio=1.0):
            self.prev_num_steps = 0
            self.num_steps = 0
            self.min_lr_ratio = min_lr_ratio
            self.max_lr_ratio = 1.0
            self.warmup_ratio = warmup_ratio
            self.stage_lr_ratio = stage_lr_ratio
            
        def set_num_steps(self, num_steps):
            self.prev_num_steps += self.num_steps
            self.num_steps = num_steps
            self.max_lr_ratio *= self.stage_lr_ratio
            logger.info(f"Setting num steps: {self.num_steps}, prev num steps: {self.prev_num_steps}, warmup ratio: {self.warmup_ratio}, min lr ratio: {self.min_lr_ratio}")
                
        def get_lr(self, batch):
            batch -= self.prev_num_steps
            warmup_steps = int(self.num_steps * self.warmup_ratio)
            try:
                if batch < warmup_steps:
                    return (batch / warmup_steps * (1.0 - self.min_lr_ratio) + self.min_lr_ratio) * self.max_lr_ratio
                else:
                    return ((1.0 - (batch - warmup_steps) / (self.num_steps - warmup_steps)) \
                            * (1.0 - self.min_lr_ratio) + self.min_lr_ratio) * self.max_lr_ratio
            except ZeroDivisionError as e:
                logger.error(f"ZeroDivisionError in LambdaLRFunc with num_steps: {self.num_steps}, warmup_ratio: {self.warmup_ratio}")
                return self.min_lr_ratio  # Trả về min_lr_ratio nếu num_steps là 0 hoặc warmup_ratio là 0
                
    lambda_lr_func = LambdaLRFunc(args.min_lr_ratio, args.warmup_ratio, args.stage_lr_ratio)
    
    if args.scheduler == 'lambda':
        if args.sheduler_type == 'epoch':
            scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda_wapper)
            logger.info(f"Using LambdaLR with epoch-based warmup, warmup epochs: {args.warmup_ep}, min lr ratio: {args.min_lr_ratio}")
        elif args.sheduler_type == 'batch':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr_func.get_lr)
            logger.info(f"Using LambdaLR with batch-based warmup, warmup ratio: {args.warmup_ratio}, min lr ratio: {args.min_lr_ratio}")
        else:
            logger.error(f"Unknown lambda type: {args.sheduler_type}, using default epoch-based warmup")
            scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda_wapper)
            
    elif args.scheduler == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=args.lr*args.min_lr_ratio, max_lr=args.lr, mode='triangular2', step_size_up=args.step_size)
        logger.info(f"Using CyclicLR with base lr: {args.lr*args.min_lr_ratio} and max lr: {args.lr}")
        
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gammalr) # TODO: Hyper parameters
        logger.info(f"Using StepLR with step size: {args.step_size} and gamma: {args.gammalr}")
        
    else:
        logger.error(f"Unknown scheduler: {args.scheduler}, using default StepLR")
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gammalr)
                  
                
    if args.parallel == 'DDP':
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=local_rank, world_size=args.world_size)
        # device_id = [0, 1, 2, 3, 4, 5, 6, 7]
        model = DDP(model, device_ids= [local_rank], find_unused_parameters=True)
    elif args.parallel == 'DP':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7' 
        model = nn.DataParallel(model, device_ids=[int(it) for it in args.device_ids.split(" ")])


    # optimizer = SGD(model.parameters(), lr=args.lr) # TODO: Use AdamW, GPU out of memory

    criterion_ce = nn.CrossEntropyLoss()
    criterion_fd = nn.CosineEmbeddingLoss()
    all_labels = []
    all_labels = list(set([t for stream in streams_indexed for t in stream if t not in all_labels]))
    task_idx = [i for i in range(len(streams_indexed))]
    labels = all_labels.copy()

    # training process
    learned_types = [0]
    prev_learned_types = [0]
    dev_scores_ls = []
    
    # Tạo class dùng để lưu old sample từ task trước
    exemplars = Exemplars(args) # TODO: 
    if args.cresume:
        logger.info(f"Resuming from {args.cresume}")
        state_dict = torch.load(args.cresume)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        task_idx = task_idx[state_dict['stage']:]
        # TODO: test use
        labels = state_dict['labels']
        learned_types = state_dict['learned_types']
        prev_learned_types = state_dict['prev_learned_types']
        
    
    e_pth = "./outputs/early_stop/" + args.log_name + ".pth"
    os.makedirs(os.path.dirname(e_pth), exist_ok=True)
    
    def _print_num_choose(num_choose):
        if args.mole_num_general_expert == 0:
            percent = num_choose / num_choose.sum(1, keepdim=True) * 100
        else:
            percent = num_choose / num_choose[:, -1].unsqueeze(1) * 100

        # Làm tròn đến 2 chữ số thập phân
        percent = torch.round(percent * 100) / 100
        percent_np = percent.cpu().numpy()

        # Tạo header cột: Expert 1, Expert 2, ...
        headers = [f"Exp {i+1}" for i in range(model.num_experts)]

        # Độ rộng cột cố định để căn đều
        col_width = 7

        # Chuẩn bị header string
        header_str = "".join(f"{h:>{col_width}}" for h in headers)

        # Chuẩn bị từng dòng dữ liệu đã căn đều, 2 chữ số thập phân
        rows_str = "\n".join(
            "".join(f"{x:>{col_width}.2f}" for x in row)
            for row in percent_np
        )

        # In ra
        logger.info(f"Num choose (percent):\n{header_str}\n{rows_str}")


    # Xét từng task 
    for stage in task_idx:
        # if stage > 0:
        #     break
        logger.info(f"==================== Stage {stage} ====================")
        
        # stage = 1 # TODO: test use
        # exemplars = Exemplars() # TODO: test use
        if args.single_label:
            stream_dataset = collect_sldataset(args.dataset, args.data_root, 'train', label2idx, stage, streams[stage], args)
        else:
            stream_dataset = collect_dataset(args.dataset, args.data_root, 'train', label2idx, stage, [i for item in streams[stage:] for i in item], args)
        if args.parallel == 'DDP':
            stream_sampler = DistributedSampler(stream_dataset, shuffle=True)
            org_loader = DataLoader(
                dataset=stream_dataset,
                sampler=stream_sampler,
                batch_size=args.batch_size,
                # batch_size=args.shot_num + int(args.class_num / args.shot_num),
                collate_fn= lambda x:x)
        else:
            org_loader = DataLoader(
                dataset=stream_dataset,
                shuffle=True,
                batch_size=args.batch_size,
                # batch_size=args.shot_num + int(args.class_num / args.shot_num),
                collate_fn= lambda x:x)
            
        stage_loader = org_loader
        if stage > 0:
            # if args.early_stop and no_better >= args.patience:
                # logger.info("Early stopping finished, loading stage: " + str(stage))
                # model.load_state_dict(torch.load(e_pth))
                
            prev_model = deepcopy(model) # TODO:test use
            for item in streams_indexed[stage - 1]:
                if not item in prev_learned_types:
                    prev_learned_types.append(item)

            logger.info(f'Loading train instances without negative instances for stage {stage}')
            
            # Lấy ra các sample để học cho task hiện tại
            exemplar_dataset = collect_exemplar_dataset(args.dataset, args.data_root, 'train', label2idx, stage-1, streams[stage-1], args)
            exemplar_loader = DataLoader(
                dataset=exemplar_dataset,
                batch_size=64,
                shuffle=True,
                collate_fn=lambda x:x)
            
            
            # Thực hiện lấy ra các sample từ class trước
            exemplars.set_exemplars(prev_model, exemplar_loader, len(learned_types), device)
            # if not args.replay:
            if not args.no_replay:
                stage_loader = exemplars.build_stage_loader(stream_dataset)
            # else:
            #     e_loader = list(exemplars.build_stage_loader(MAVEN_Dataset([], [], [], [])))
            if args.rep_aug != "none":
                e_loader = exemplars.build_stage_loader(MAVEN_Dataset([], [], [], [], []))
            # prev_model.to(args.device)   # TODO: test use

        for item in streams_indexed[stage]:
            if not item in learned_types:
                learned_types.append(item)
        logger.info(f'Learned types: {learned_types}')
        logger.info(f'Previous learned types: {prev_learned_types}')
        
        labels_all_learned_types = [idx2label[x] for x in learned_types]
        
        description_stage_loader = DataLoader(
            DescriptionDataset(args, tokenizer, labels_all_learned_types),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_description
        )
        
        dev_score = None
        no_better = 0

        if stage > 0:
            ep_time = args.task_ep_time
        else:
            ep_time = 1
            
        num_epochs = int(args.epochs * ep_time)
        
        lambda_lr_func.set_num_steps(len(stage_loader) * num_epochs)
        logger.info("Start training ...")
        for ep in tqdm(range(num_epochs), desc="Epoch"):
            if stage == 0 and args.skip_first:
                continue
            
            if args.use_mole:
                if ep < args.uniform_ep:
                    model.turn_uniform_expert(turn_on=True)
                else:
                    model.turn_uniform_expert(turn_on=False)
                    
            model.train()
            if args.gradient_checkpointing:
                # model.gradient_checkpointing_enable()
                pass
            
            wandb.log({"epoch": ep + 1 + (num_epochs * (stage - 1) + args.epochs if stage > 0 else 0), "stage": stage})
            
            iter_cnt = 0
            for batch in stage_loader:
                iter_cnt += 1
                optimizer.zero_grad()
                
                train_x, train_y, train_masks, train_span, train_augment = zip(*batch)
                train_x = torch.LongTensor(train_x).to(device)
                train_masks = torch.LongTensor(train_masks).to(device)
                train_y = [torch.LongTensor(item).to(device) for item in train_y]           
                train_span = [torch.LongTensor(item).to(device) for item in train_span]     # Sử dụng để lưu vị trí bắt đầu và kết thúc 1 từ của các ids
                augment_x = {}
                augment_masks = {}
                augment_y = {}
                augment_span = {}
                for aug_ids in range(args.num_augmention):
                    augment_x[aug_ids] = [torch.LongTensor(item[aug_ids][0]).to(device) for item in train_augment]
                    augment_y[aug_ids] = [torch.LongTensor(item[aug_ids][1]).to(device) for item in train_augment]
                    augment_masks[aug_ids] = [torch.LongTensor(item[aug_ids][2]).to(device) for item in train_augment]
                    augment_span[aug_ids] = [torch.LongTensor(item[aug_ids][3]).to(device) for item in train_augment]

                augment_x_list = [
                    torch.stack(value, dim=0)  # → (B, L)
                    for _, value in augment_x.items()
                ]
                augment_x_total = torch.cat(augment_x_list, dim=0).to(device)

                augment_y_total = [
                    tensor
                    for value in augment_y.values()
                    for tensor in value
                ]
                
                augment_masks_list = [
                    torch.stack(value, dim=0)  # → (B, L)
                    for _, value in augment_masks.items()
                ]
                augment_masks_total = torch.cat(augment_masks_list, dim=0).to(device)
                
                augment_span_total = [
                    tensor
                    for value in augment_span.values()
                    for tensor in value
                ]
                
                labels_for_loss_des = []
                for y in train_y:
                    for k in y:
                        if k in learned_types and k != 0:
                           labels_for_loss_des.append(idx2label[int(k)])
                           break 
                # print(labels)
                
                # if args.dataset == "ACE":
                #     return_dict = model(train_x, train_masks)
                # else: 
                return_dict = model(train_x, train_masks, train_span)
                outputs, context_feat, trig_feat = return_dict['outputs'], return_dict['context_feat'], return_dict['trig_feat']
                logits_router = return_dict['logits_router']

                # invalid_mask_op = torch.BoolTensor([item not in learned_types for item in range(args.class_num)]).to(device)
                # not from below's codes
                
                # Loại bỏ ra những sample có label không được học trong term này
                for i in range(len(train_y)):
                    invalid_mask_label = torch.BoolTensor([item not in learned_types for item in train_y[i]]).to(device)
                    train_y[i].masked_fill_(invalid_mask_label, 0)
                # outputs[:, 0] = 0
                loss, loss_ucl, loss_aug, loss_fd, loss_pd, loss_tlcl = 0, 0, 0, 0, 0, 0
                ce_y = torch.cat(train_y) # (sum of len(label), )
                ce_outputs = outputs
                if (args.ucl or args.tlcl) and (stage > 0 or (args.skip_first_cl != "ucl+tlcl" and stage == 0)):                        

                    reps = return_dict['reps']
                    bs, hdim = reps.shape
                    aug_repeat_times = args.aug_repeat_times
                    # Tạo data augment
                    da_x = train_x.clone().repeat((aug_repeat_times, 1))
                    da_y = train_y * aug_repeat_times
                    da_masks = train_masks.repeat((aug_repeat_times, 1))
                    da_span = train_span * aug_repeat_times
                    tk_len = torch.count_nonzero(da_masks, dim=-1) - 2
                    # Thực hiện hoán vị random các vị trí cho các câu được augment
                    perm = [torch.randperm(item).to(device) + 1 for item in tk_len]
                    
                    # Thực hiện augment cho câu
                    
                    # Thực hiện hoán đổi vị trí của từ trong câu theo perm
                    if args.cl_aug == "shuffle":
                        for i in range(len(tk_len)):
                            da_span[i] = torch.where(da_span[i].unsqueeze(2) == perm[i].unsqueeze(0).unsqueeze(0))[2].view(-1, 2) + 1
                            da_x[i, 1: 1+tk_len[i]] = da_x[i, perm[i]]
                    # Thực hiện như trên nhưng chỉ 25% số câu được augment
                    elif args.cl_aug =="RTR":
                        rand_ratio = 0.25
                        rand_num = (rand_ratio * tk_len).int()
                        
                        # Các token được chọn ra để không hoán đổi
                        special_ids = [103, 102, 101, 100, 0]
                        all_ids = torch.arange(model.backbone.config.vocab_size).to(device)
                        special_token_mask = torch.ones(model.backbone.config.vocab_size).to(device)
                        special_token_mask[special_ids] = 0
                        all_tokens = all_ids.index_select(0, special_token_mask.nonzero().squeeze())
                        for i in range(len(rand_num)):
                            token_idx = torch.arange(tk_len[i]).to(device) + 1
                            trig_mask = torch.ones(token_idx.shape).to(device)
                            if args.dataset == "ACE":
                                span_pos = da_span[i][da_y[i].nonzero()].view(-1).unique() - 1
                            else:
                                span_pos = da_span[i].view(-1).unique() - 1
                            # Các token được chọn ra để không hoán đổi 
                            trig_mask[span_pos] = 0
                            token_idx_ntrig = token_idx.index_select(0, trig_mask.nonzero().squeeze())
                            replace_perm = torch.randperm(token_idx_ntrig.shape.numel())
                            replace_idx = token_idx_ntrig[replace_perm][:rand_num[i]]
                            new_tkn_idx = torch.randperm(len(all_tokens))[:rand_num[i]]
                            da_x[i, replace_idx] = all_tokens[new_tkn_idx].to(device)
                    # if args.dataset == "ACE":
                    #     da_return_dict = model(da_x, da_masks)
                    # else:
                    
                    # Hidden representaion của data augment
                    da_return_dict = model(da_x, da_masks, da_span)
                    da_outputs, da_reps, da_context_feat, da_trig_feat = da_return_dict['outputs'], da_return_dict['reps'], da_return_dict['context_feat'], da_return_dict['trig_feat']
                    da_logits_router = da_return_dict['logits_router']
                    
                    # Contrastive loss cho sentence
                    if args.ucl:
                        if not ((args.skip_first_cl == "ucl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                            ucl_reps = torch.cat([reps, da_reps])
                            ucl_reps = normalize(ucl_reps, dim=-1)
                            Adj_mask_ucl = torch.zeros(bs * (1 + aug_repeat_times), bs * (1 + aug_repeat_times)).to(device)
                            for i in range(aug_repeat_times):
                                Adj_mask_ucl += torch.eye(bs * (1 + aug_repeat_times)).to(device)
                                Adj_mask_ucl = torch.roll(Adj_mask_ucl, bs, -1)                    
                            loss_ucl = compute_CLLoss(Adj_mask_ucl, ucl_reps, bs * (1 + aug_repeat_times), args, device)
                            
                    # Contrastive loss cho trigger
                    if args.tlcl:
                        if not ((args.skip_first_cl == "tlcl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                            tlcl_feature = torch.cat([trig_feat, da_trig_feat])
                            # tlcl_feature = trig_feat
                            tlcl_feature = normalize(tlcl_feature, dim=-1)
                            tlcl_lbs = torch.cat(train_y + da_y)
                            # tlcl_lbs = torch.cat(train_y)
                            mat_size = tlcl_feature.shape[0]
                            tlcl_lbs_oh = F.one_hot(tlcl_lbs).float()
                            # tlcl_lbs_oh[:, 0] = 0 # whether to compute negative distance
                            Adj_mask_tlcl = torch.matmul(tlcl_lbs_oh, tlcl_lbs_oh.T)
                            Adj_mask_tlcl = Adj_mask_tlcl * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                            loss_tlcl = compute_CLLoss(Adj_mask_tlcl, tlcl_feature, mat_size, args, device)
                    # loss = loss + loss_ucl + loss_tlcl*args.weight_loss_tlcl
                    if args.joint_da_loss == "ce" or args.joint_da_loss == "mul":
                        ce_y = torch.cat(train_y + da_y)
                        ce_outputs = torch.cat([outputs, da_outputs])


                
                    # outputs[i].masked_fill_(invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                # if args.dataset == "ACE":
                loss_des_cl = torch.tensor(0.0, device=device)
                if args.use_description:
                    
                    reps = trig_feat
                    descriptions_representations = {}
                    final_description_res = {}
                    
                    model.eval()
                    with torch.no_grad():
                        for bt, description_batch in enumerate(description_stage_loader):

                            train_x_description, train_masks_description, keys = description_batch
                            train_x_description = torch.LongTensor(train_x_description).to(device)
                            train_masks_description = torch.LongTensor(train_masks_description).to(device)
                
                            return_dict_description = model.forward_cls(train_x_description, train_masks_description)
                            context_feat_descriptions = return_dict_description
                            for key, context_feat_description in zip(keys, context_feat_descriptions):
                                if key not in descriptions_representations:
                                    descriptions_representations[key] = []
                                descriptions_representations[key].append(context_feat_description)
                                                                
                        for key, value in descriptions_representations.items():
                            feature = torch.stack(value, dim=0)
                            temp = torch.mean(feature, dim=0)
                            final_description_res[key] = temp
                            
                    model.train()
                    
                    if args.loss_des_type == "1":
                        negative_dict = find_negative_labels(final_description_res)       
                        loss_des_cl = contrastive_loss_des(reps, labels_for_loss_des, final_description_res, negative_dict)       
        
                    elif args.loss_des_type == "2":
                        des_feat = []
                        des_y = []
                        for key_des, value_des in final_description_res.items():
                            des_feat.append(value_des)
                            des_y.append(key_des)
                        
                        des_feat = torch.stack(des_feat, dim=0) 
                        des_y_tensor = torch.tensor([label2idx[int(xx)] for xx in des_y],
                                                    dtype=torch.long,
                                                    device=device)
                        
                        
                        des_cl_feature = torch.cat([trig_feat, des_feat])
                        # tlcl_feature = trig_feat
                        des_cl_feature = normalize(des_cl_feature, dim=-1)
                        des_cl_lbs = torch.cat(train_y + [des_y_tensor], dim=0)
                        # tlcl_lbs = torch.cat(train_y)
                        des_mat_size = des_cl_feature.shape[0]
                        des_cl_lbs_oh = F.one_hot(des_cl_lbs).float()
                        # tlcl_lbs_oh[:, 0] = 0 # whether to compute negative distance
                        Des_adj_mask_tlcl = torch.matmul(des_cl_lbs_oh, des_cl_lbs_oh.T)
                        Des_adj_mask_tlcl = Des_adj_mask_tlcl * (torch.ones(des_mat_size) - torch.eye(des_mat_size)).to(device)
                        loss_des_cl = compute_CLLoss(Des_adj_mask_tlcl, des_cl_feature, des_mat_size, args, device)
                    
                    # loss = loss + loss_des_cl * args.ratio_loss_des_cl      
                    
                lgacl_loss = torch.tensor(0.0, device=device)
                if args.gpt_augmention:
                    augment_return_dict = model(augment_x_total, augment_masks_total, augment_span_total)
                    augment_trig_feat = augment_return_dict['trig_feat']
                    
                    lgacl_feature = torch.cat([trig_feat, augment_trig_feat])
                    # tlcl_feature = trig_feat
                    lgacl_feature = normalize(lgacl_feature, dim=-1)
                    lgacl_lbs = torch.cat(train_y + augment_y_total, dim=0)
                    if args.decrease_0_gpt_augmention:
                        lgacl_feature, lgacl_lbs = balance_zero_with_nonzero(lgacl_feature, lgacl_lbs, args)
                    # tlcl_lbs = torch.cat(train_y)
                    mat_size = lgacl_feature.shape[0]
                    lgacl_lbs_oh = F.one_hot(lgacl_lbs).float()
                    # tlcl_lbs_oh[:, 0] = 0 # whether to compute negative distance
                    Adj_mask_lgacl = torch.matmul(lgacl_lbs_oh, lgacl_lbs_oh.T)
                    Adj_mask_lgacl = Adj_mask_lgacl * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                    lgacl_loss = compute_CLLoss(Adj_mask_lgacl, lgacl_feature, mat_size, args, device)
                else:
                    lgacl_loss = torch.tensor(0.0, device=device)
                
                # loss = loss + lgacl_loss * args.ratio_loss_lgacl
                
                # Loss ce cho class ở task hiện tại
                ce_outputs = ce_outputs[:, learned_types]
                if args.use_weight_ce:
                    loss_ce = CrossEntropyLossWithWeight(ce_outputs, ce_y, alpha=args.alpha_ce)
                else:
                    loss_ce = criterion_ce(ce_outputs, ce_y)
                # loss = loss + loss_ce
                w = len(prev_learned_types) / len(learned_types)

                # Loss ce cho class ở task cũ

                if args.rep_aug != "none" and stage > 0:
                    outputs_aug, aug_y = [], []
                    for e_batch in e_loader:
                        exemplar_x, exemplars_y, exemplar_masks, exemplar_span, exemplar_augment = zip(*e_batch)
                        exemplar_radius = [exemplars.radius[y[0]] for y in exemplars_y]
                        exemplar_x = torch.LongTensor(exemplar_x).to(device)
                        exemplar_masks = torch.LongTensor(exemplar_masks).to(device)
                        exemplars_y = [torch.LongTensor(item).to(device) for item in exemplars_y]
                        exemplar_span = [torch.LongTensor(item).to(device) for item in exemplar_span]    
                        augment_exemplars_x = {}
                        augment_exemplars_masks = {}
                        augment_exemplars_y = {}
                        augment_exemplars_span = {}
                        for aug_ids in range(args.num_augmention):
                            augment_exemplars_x[aug_ids] = [torch.LongTensor(item[aug_ids][0]).to(device) for item in exemplar_augment]
                            augment_exemplars_y[aug_ids] = [torch.LongTensor(item[aug_ids][1]).to(device) for item in exemplar_augment]
                            augment_exemplars_masks[aug_ids] = [torch.LongTensor(item[aug_ids][2]).to(device) for item in exemplar_augment]
                            augment_exemplars_span[aug_ids] = [torch.LongTensor(item[aug_ids][3]).to(device) for item in exemplar_augment]
                                    
                        if args.rep_aug == "relative":
                            aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(exemplar_radius)).unsqueeze(-1))
                        else:
                            aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(list(exemplars.radius.values())).mean()))
                        output_aug = aug_return_dict['outputs_aug']
                        outputs_aug.append(output_aug)
                        aug_y.extend(exemplars_y)
                    outputs_aug = torch.cat(outputs_aug)
                    if args.leave_zero:
                        outputs_aug[:, 0] = 0
                    outputs_aug = outputs_aug[:, learned_types].squeeze(-1)
                    loss_aug = criterion_ce(outputs_aug, torch.cat(aug_y))
                    # loss = loss_ce * w + loss_aug * (1 - w)
                    # loss = loss_ce * (1 - w) + loss_aug * w
                    # loss = args.gamma * loss + args.theta * loss_aug
                
                    
                # Loss distill của previous model cho current model nhằm giữ lại kiến thức cũ từ mô hình cũ. ( Không dùng đến trong bài )
                if stage > 0:
                    prev_model.eval()
                    with torch.no_grad():
                        prev_return_dict = prev_model(train_x, train_masks, train_span)
                        prev_outputs, prev_feature = prev_return_dict['outputs'], prev_return_dict['context_feat']
                        prev_logits_router = prev_return_dict['logits_router']
                        

                        if args.joint_da_loss == "dist" or args.joint_da_loss == "mul":
                            outputs = torch.cat([outputs, da_outputs])
                            context_feat = torch.cat([context_feat, da_context_feat])
                            logits_router = torch.cat([logits_router, da_logits_router])
                            prev_return_dict_cl = prev_model(da_x, da_masks, da_span)
                            prev_outputs_cl, prev_feature_cl = prev_return_dict_cl['outputs'], prev_return_dict_cl['context_feat']
                            prev_logits_router_cl = prev_return_dict_cl['logits_router']
                            prev_outputs, prev_feature = torch.cat([prev_outputs, prev_outputs_cl]), torch.cat([prev_feature, prev_feature_cl])
                            prev_logits_router = torch.cat([prev_logits_router, prev_logits_router_cl])
                    # prev_invalid_mask_op = torch.BoolTensor([item not in prev_learned_types for item in range(args.class_num)]).to(device)
                    prev_valid_mask_op = torch.nonzero(torch.BoolTensor([item in prev_learned_types for item in range(args.class_num + 1)]).to(device))
                    if "fd" in args.distill or "mul" in args.distill:
                        prev_feature = normalize(prev_feature.view(-1, prev_feature.shape[-1]), dim=-1)
                        cur_feature = normalize(context_feat.view(-1, prev_feature.shape[-1]), dim=-1)
                        loss_fd = criterion_fd(prev_feature, cur_feature, torch.ones(prev_feature.size(0)).to(device)) # TODO: Don't know whether the code is right
                    else:
                        loss_fd = 0

                    if "pd" in args.distill or "mul" in args.distill:
                        T = args.temperature
                        if args.leave_zero:
                            prev_outputs[:, 0] = 0
                        prev_outputs = prev_outputs[:, prev_valid_mask_op].squeeze(-1)
                        cur_outputs = outputs[:, prev_valid_mask_op].squeeze(-1)
                        prev_p = torch.softmax(prev_outputs / T, dim= -1)
                        p = torch.log_softmax(cur_outputs / T, dim = -1)
                        loss_pd = -torch.mean(torch.sum(prev_p * p, dim = -1), dim = 0)
                    else:
                        loss_pd = 0
                        
                    if "rd" in args.distill or "mul" in args.distill:
                        # Flatten từ [num_layer, num_token, num_experts] => [num_layer * num_token, num_experts]
                        prev_flat = prev_logits_router.view(-1, prev_logits_router.shape[-1])
                        cur_flat  = logits_router.view(-1, logits_router.shape[-1])

                        # Tạo target cho cosine embedding loss (1 nghĩa là muốn chúng giống nhau)
                        target = torch.ones(prev_flat.size(0)).to(device)

                        # Tính loss distill cho logits_router
                        loss_rd = criterion_fd(prev_flat, cur_flat, target).to(device)
                    else:
                        loss_rd = 0
                
                else:
                    loss_fd = 0
                    loss_pd = 0
                    loss_rd = 0

                    
                    # if args.dweight_loss and stage > 0:
                    #     loss = loss * (1 - w) + (loss_fd + loss_pd + loss_rd) * w
                    # else:
                    #     loss = loss + args.alpha * loss_fd + args.beta * loss_pd

                loss = loss_ce + loss_ucl * args.ratio_loss_ucl + \
                        loss_tlcl * args.ratio_loss_tlcl + loss_des_cl * args.ratio_loss_des_cl + \
                        loss_aug * args.ratio_loss_aug + lgacl_loss * args.ratio_loss_lgacl + \
                        loss_fd * args.ratio_loss_fd + loss_pd * args.ratio_loss_pd + loss_rd * args.ratio_loss_rd
                loss.backward()
                total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if args.print_trainable_params:
                    # model.print_trainable_parameters()
                    pass
                
                optimizer.step() 
                model.tune_bias()
                model.clear_num_choose()
                if args.sheduler_type == 'batch':
                    scheduler.step()
                # stats = torch.cuda.memory_stats()
                wandb.log({      
                            # "memory/allocated_MB": stats["allocated_bytes.all.current"] / 1024**2,
                            # "memory/allocated_peak_MB": stats["allocated_bytes.all.peak"] / 1024**2,
                            # "memory/reserved_MB": stats["reserved_bytes.all.current"] / 1024**2,
                            # "memory/reserved_peak_MB": stats["reserved_bytes.all.peak"] / 1024**2,
                            # "memory/active_MB": stats["active_bytes.all.current"] / 1024**2,
                            # "memory/num_ooms": stats["num_ooms"],
                            # "memory/alloc_retries": stats["num_alloc_retries"],
                            
                            "loss_ucl": loss_ucl,
                            # "loss_tlcl": loss_tlcl,
                            "loss_all": loss,
                            "loss_ce": loss_ce,
                            "loss_aug": loss_aug,
                            "loss_des_cl": loss_des_cl,
                            "loss_lgacl": lgacl_loss,
                            "loss_fd": loss_fd,
                            "loss_pd": loss_pd,
                            "loss_rd": loss_rd,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "total_norm": total_norm,
                        })
                
            
            if args.sheduler_type == 'epoch':
                scheduler.step()
    
            if ((ep + 1) % max(int(args.eval_freq*ep_time), 1) == 0 and args.early_stop and ((ep + 1) >= args.skip_eval_ep*ep_time or stage > 0)) or (ep + 1) == num_epochs: # TODO TODO
                # Evaluation process
                logger.info("Evaluation process ...")
                model.eval()
                with torch.no_grad():
                    if args.single_label:
                        eval_dataset = collect_eval_sldataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item], args)
                    else:
                        eval_dataset = collect_dataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item], args)
                    eval_loader = DataLoader(
                        dataset=eval_dataset,
                        shuffle=True,
                        batch_size=args.eval_batch_size,
                        collate_fn=lambda x:x)
                    calcs = Calculator()

                    for batch in tqdm(eval_loader, desc="Eval"):
                        eval_x, eval_y, eval_masks, eval_span = zip(*batch)
                        eval_x = torch.LongTensor(eval_x).to(device)
                        eval_masks = torch.LongTensor(eval_masks).to(device)
                        eval_y = [torch.LongTensor(item).to(device) for item in eval_y]
                        eval_span = [torch.LongTensor(item).to(device) for item in eval_span]  
                        eval_return_dict = model(eval_x, eval_masks, eval_span, train=False)

                        eval_outputs = eval_return_dict['outputs']
                        valid_mask_eval_op = torch.BoolTensor([idx in learned_types for idx in range(args.class_num + 1)]).to(device)
                        for i in range(len(eval_y)):
                            invalid_mask_eval_label = torch.BoolTensor([item not in learned_types for item in eval_y[i]]).to(device)
                            eval_y[i].masked_fill_(invalid_mask_eval_label, 0)
                        if args.leave_zero:
                            eval_outputs[:, 0] = 0
                        eval_outputs = eval_outputs[:, valid_mask_eval_op].squeeze(-1)
                        calcs.extend(eval_outputs.argmax(-1), torch.cat(eval_y))
                        
                    bc, (precision, recall, micro_F1) = calcs.by_class(learned_types)
                    wandb.log({
                        f"precision": precision,
                        f"recall": recall,
                        f"micro_F1": micro_F1,
                    })

                    
                    logger.info(f'marco F1 {micro_F1}')
                    logger.info(f"bc:{bc}")
                    num_choose = model.get_num_choose()
                    model.clear_num_choose()
                    
                    # report to optuna
                    # if args.early_stop:
                    #     if dev_score is None or dev_score < micro_F1:
                    #         no_better = 0
                    #         dev_score = micro_F1
                    #         torch.save(model.state_dict(), e_pth)
                    #         if stage == 0:
                    #             torch.save(model.backbone.state_dict(), "outputs/best_model0.pth")
                    #     else:
                    #         no_better += 1
                    #         logger.info(f'No better: {no_better}/{args.patience}')
                    #     if no_better >= args.patience:
                    #         _print_num_choose(num_choose)
                    #         logger.info("Early stopping with dev_score: " + str(dev_score))
                    #         dev_scores_ls.append(dev_score)
                    #         logger.info(f"Dev scores list: {dev_scores_ls}")
                    #         break
                    
                    if ep + 1 == num_epochs:
                        _print_num_choose(num_choose)
                        # if args.early_stop:
                        #     logger.info("Early stopping with dev_score: " + str(dev_score))
                        #     dev_scores_ls.append(dev_score)
                        #     logger.info(f"Dev scores list: {dev_scores_ls}")
     
                        # else:
                        #     logger.info("Final model with dev_score: " + str(micro_F1))
                        #     dev_scores_ls.append(micro_F1)
                        #     logger.info(f"Dev scores list: {dev_scores_ls}")
                            
                        logger.info("Final model with dev_score: " + str(micro_F1))
                        dev_scores_ls.append(micro_F1)
                        logger.info(f"Dev scores list: {dev_scores_ls}")
                    
                    # if trial is not None:
                    #     trial.report(micro_F1, ep + 1 + args.epochs * stage)
                    #     if trial.should_prune():
                    #         raise optuna.TrialPruned()
                        
                        
        for tp in streams_indexed[stage]:
            if not tp == 0:
                labels.pop(labels.index(tp))
                
        if args.save_dir and local_rank == 0:
            save_stage = stage + 1
            state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'stage':save_stage, 
                            'labels':labels, 'learned_types':learned_types, 'prev_learned_types':prev_learned_types}
            save_pth = os.path.join(args.save_dir, "perm" + str(args.perm_id))
            cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            save_name = f"stage_{save_stage}_{cur_time}.pth"
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            logger.info(f'state_dict saved to: {os.path.join(save_pth, save_name)}')
            torch.save(state, os.path.join(save_pth, save_name))
            os.remove(e_pth)
            logger.info(f"Best model saved to {save_pth}")
            
    
    wandb.finish()
    
    return np.mean(dev_scores_ls)