import torch
from torch.optim.lr_scheduler import MultiStepLR
from transformers import AdamW
from transformers import Trainer

# https://discuss.huggingface.co/t/how-do-use-lr-scheduler/4046
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom attributes here
            
def create_optimizer_and_scheduler(self, num_training_steps: int):
    # no_decay = ["bias", "LayerNorm.weight"]
    # # Add any new parameters to optimize for here as a new dict in the list of dicts
    # optimizer_grouped_parameters = ...   

    # self.optimizer = AdamW(optimizer_grouped_parameters, 
    #                     lr=args.lr,
    #                     betas=(training_args.adam_beta1, training_args.adam_beta2),
    #                     eps=training_args.adam_epsilon
    #                     )
    # self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=args.milestones, gamma=args.gamma)    

    self.optimmizer, self.scheduler = self.optimizers

    