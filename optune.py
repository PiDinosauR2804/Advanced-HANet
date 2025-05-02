from classifier.train import train
import optuna
import wandb

def objective(trial):
    args = {
        'perm_id': 0,
        'dataset': 'MAVEN',
        'stream_root': './data/data_ids',
        'max_seqlen': 120,
        'adamw_eps': 1e-7,
        'fixed_enum': True,
        'enum': 1,
        'temperature': 2,
        'task_num': 5,
        'early_stop': False,
        'patience': 5,
        'eval_freq': 1,
        'input_map': True,
        'class_num': 10,
        'shot_num': 5,
        'e_weight': 50,
        'no_replay': False,
        'period': 10,
        'epochs': trial.suggest_int('epochs', 10, 30),
        'batch_size': trial.suggest_int('batch_size', 4, 16),
        'device': "cuda:2",
        'log': True, 
        'log_name': f"optuna_{trial.number}",
        'data_root': './data_incremental',
        'backbone': trial.suggest_categorical('backbone', ['bert-base-uncased', 'roberta-base']),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-3),
        'decay': trial.suggest_loguniform('decay', 1e-5, 1e-3),
        # Add other hyperparameters as needed
    }
    
    wandb.init(project="optuna-mnist", config=args, reinit=True)
    f1 = train(0, args, trial)
    wandb.finish()
    
    return f1

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))