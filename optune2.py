from classifier.train import train
from configs import parse_arguments
import optuna
import wandb
import torch
from loguru import logger
import os 
os.environ['WANDB_API_KEY'] = 'bbee5bd41b9c06ce3048243c9611e36701652ef2'  # Đặt API key của bạn ở đây

args = parse_arguments()

def objective(trial):
    # Hằng 
    args.epochs = 10
    args.logs_dir = "logs/classifier"
    args.data_root = "./data/data_ids_enhence"
    args.dataset = "ACE"
    args.backbone = "bert-base-uncased"
    args.decay = 1e-4
    args.no_freeze_bert = True
    args.shot_num = 5
    args.device = "cuda:0"
    args.log = True
    args.log_dir = "./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/"
    args.log_name = "ashuffle_lnone_r1"
    args.dweight_loss = True
    args.rep_aug = "mean"
    args.distill = "pd"
    args.class_num = 10
    args.single_label = True
    args.cl_aug = "shuffle"
    args.aug_repeat_times = 3
    args.joint_da_loss = "ce"
    args.sub_max = True
    args.cl_temp = 0.07
    args.tlcl = False
    args.ucl = True
    args.skip_first_cl = "ucl+tlcl"
    args.use_description = True
    args.num_description = 3
    args.ratio_loss_des_cl = 1
    args.task_ep_time = 1
    args.early_stop = True
    args.skip_eval_ep = 0
    args.eval_freq = 1
    args.patience = 5
    args.early_stop = True
    args.classifier_layer = 1
    args.hidden_dim = 128
    args.dropout = 0.5
    args.use_mole = True
    args.step_size = 1
    args.wandb = True
    args.eval_batch_size = 64
    args.task_ep_time = 1
    args.uniform_ep = 3
    args.lora_dropout = 0.3
    args.gpt_augmention = True
    args.use_weight_ce = True
    args.entropy_weight = 0.1
    args.load_balance_weight = 1
    args.general_expert_weight = 0.2
    args.batch_size = 4
    args.project_name = "HANet_new_mole_ace_2_5"

    # Tham số cho Optuna trial
    args.lora_rank = trial.suggest_int("lora_rank", 64, 256, step=64)
    args.lora_alpha = trial.suggest_int("lora_alpha", 64, 256, step=64)
    args.mole_num_experts = trial.suggest_categorical("mole_num_experts", [4, 8])
    args.mole_top_k = trial.suggest_categorical("mole_top_k", [2, 4])
    args.mole_num_general_expert = trial.suggest_categorical("mole_num_general_expert", [0, 1, 2])
    
    args.lr = trial.suggest_float("lr", 2e-5, 2e-4, log=True)
    args.gammalr = trial.suggest_float("gamma", 0.9, 1.0, step=0.01)
    
    try:
        f1 = train(0, args, trial)
        return f1
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("CUDA out of memory. Releasing GPU memory...")
            torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU
            return float('inf')  # Giá trị lỗi để Optuna bỏ qua
        else:
            raise e  # Nếu là lỗi khác, vẫn cho nó raise lên


if __name__ == "__main__":
    wandb.login()
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))