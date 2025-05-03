from classifier.train import train
from configs import parse_arguments
import optuna
import wandb

args = parse_arguments()

def objective(trial):
    # Hằng 
    args.epochs = 100
    args.early_stopping = True
    args.use_lora = True
    args.logs_dir = "logs/classifier"
    args.data_root = "./data/data_ids"
    args.dataset = "MAVEN"
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
    args.distill = "mul"
    args.class_num = 20
    args.single_label = True
    args.cl_aug = "shuffle"
    args.aug_repeat_times = 5
    args.joint_da_loss = "ce"
    args.sub_max = True
    args.cl_temp = 0.07
    args.tlcl = True
    args.ucl = True
    args.skip_first_cl = "ucl+tlcl"
    args.use_description = True
    args.num_description = 3
    args.ratio_loss_des_cl = 0.1
    

    # Tham số cho Optuna trial
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.lora_rank = trial.suggest_int("lora_rank", 32, 256, step=32)
    args.lora_alpha = trial.suggest_int("lora_alpha", 16, 128, step=16)
    args.lora_dropout = trial.suggest_float("lora_dropout", 0.1, 0.3, step=0.05)
    args.task_ep_time = trial.suggest_int("task_ep_time", 1, 5)
    args.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    
    f1 = train(0, args, trial)
    
    return f1

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))