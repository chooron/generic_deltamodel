import sys

sys.path.append(r'E:\pycode\generic_deltamodel')
from dmg import ModelHandler
from dmg.core.utils import import_data_loader, import_trainer, set_randomseed
from project.better_estimate import load_config

#------------------------------------------#
# Define model settings here.
CONFIG_PATH = r'conf/config_dhbv_pub_lstm.yaml'
#------------------------------------------#
# model training
config = load_config(CONFIG_PATH)
config['mode'] = 'train'
# config['train']['start_epoch'] = 10
set_randomseed(config['random_seed'])
model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)
trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(
    config,
    model,
    train_dataset=data_loader.train_dataset,
    verbose=True
)

# trainer.train()
print(f"Training complete. Model saved to \n{config['model_path']}")

# model evaluation
config['mode'] = 'test'
config['test']['test_epoch'] = 50
set_randomseed(config['random_seed'])

model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)

trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(
    config,
    model,
    eval_dataset=data_loader.eval_dataset,
    verbose=True,
)

print('Evaluating model...')
trainer.evaluate()
print(f"Metrics and predictions saved to \n{config['out_path']}")