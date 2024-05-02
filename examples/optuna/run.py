import sys
sys.path.append('/lustre1/project/stg_00032/georgem/closure/')
import src.trainers as tr
import optuna
import pickle
import logging
logging.basicConfig(level=logging.INFO)


import copy
import numpy as np

trainer = tr.Trainer(work_dir='./')
logger = logging.getLogger('trainer')
config = copy.deepcopy(trainer.config)
request_targets = config['dataset_kwargs']['read_features_targets_kwargs']['request_targets']
request_features = config['dataset_kwargs']['read_features_targets_kwargs']['request_features']

def objective(trial):
    # Generate the model.

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024])
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1, log=True)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 5)
    num_neurons = trial.suggest_categorical("num_neurons", [20, 60, 100])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    dropouts = [dropout_rate for i in range(num_hidden_layers+1)]
    feature_dims = [len(request_features)] + [num_neurons for i in range(num_hidden_layers)] + [1]
    activations = [trial.suggest_categorical(f"activation_{i}", ["ReLU", "Tanh", "Sigmoid","ELU"]) for i in range(num_hidden_layers)] + [None]

    config['model_kwargs']['optimizer_kwargs']['weight_decay'] = weight_decay
    config['model_kwargs']['optimizer_kwargs']['lr'] = lr
    config['model_kwargs']['activations'] = activations
    config['model_kwargs']['feature_dims'] = feature_dims
    config['model_kwargs']['dropouts'] = dropouts
    config['load_data_kwargs']['train_loader_kwargs']['batch_size'] = batch_size 
    
    config['run'] = f"{targets}/{trial.number}"
    config['trial'] = trial
    loss = trainer.fit(config=config)
    logger.info(f"Trial {trial.number} : {loss}")
    return loss


for targets in request_targets:
    logger.info("<<<<<<<<<<<<<Optimizing for targets: ", targets)
    config['load_data_kwargs']['train_loader_kwargs']['target_channel_names'] = [targets]
    config['load_data_kwargs']['val_loader_kwargs']['target_channel_names'] = [targets]
    study = optuna.create_study(study_name=targets, storage= f"sqlite:///{targets}.db",direction="minimize")
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}", )
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Number: {trial.number}")
    logger.info(f"  Value: { trial.value}")

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    with open("sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)


