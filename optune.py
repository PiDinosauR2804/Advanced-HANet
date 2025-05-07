from classifier.train import train
from configs import parse_arguments
import optuna
import wandb

args = parse_arguments()

def objective(trial):
    # Hằng 
    args.epochs = 50
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
    args.distill = "pd"
    args.class_num = 20
    args.single_label = True
    args.cl_aug = "shuffle"
    args.aug_repeat_times = 5
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
    args.skip_eval_ep = 10
    args.eval_freq = 2
    args.patience = 3
    args.early_stop = True
    args.classifier_layer = 1
    args.hidden_dim = 128
    args.dropout = 0.5
    args.use_general_expert = True
    args.use_mole = True
    args.step_size = 5
    args.wandb = True
    args.eval_batch_size = 32
    args.task_ep_time = 1

    # Tham số cho Optuna trial
    args.uniform_ep = trial.suggest_int("uniform_ep", 1, 10)
    args.lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
    args.lora_rank = trial.suggest_int("lora_rank", 32, 256, step=32)
    args.lora_alpha = trial.suggest_int("lora_alpha", 16, 128, step=16)
    args.lora_dropout = trial.suggest_float("lora_dropout", 0.1, 0.5, step=0.05)
    args.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    args.mole_num_experts = trial.suggest_categorical("mole_num_experts", [4, 8])
    args.mole_top_k = trial.suggest_categorical("mole_top_k", [2, 4])
        
    args.gammalr = trial.suggest_float("gamma", 0.9, 1.0, step=0.01)
    args.entropy_weight = trial.suggest_float("entropy_weight", 0.01, 1.0, step=0.01)
    args.load_balance_weight = trial.suggest_float("load_balance_weight", 0.01, 1.0, step=0.01)
    args.general_expert_weight = trial.suggest_float("general_expert_weight", 0.1, 1.0, step=0.1)
    
    f1 = train(0, args, trial)
    
    return f1

if __name__ == "__main__":
    wandb.login()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))