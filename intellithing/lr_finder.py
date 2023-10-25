from transformers import TrainerCallback
import matplotlib.pyplot as plt
import math

class LearningRateFinderCallback(TrainerCallback):
    def __init__(self, start_lr, end_lr):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_training_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.num_training_steps = args.max_steps
        if self.num_training_steps is None:
            raise ValueError("Please specify max_steps.")
        self.lrs = []
        self.losses = []


    def on_step_end(self, args, state, control, **kwargs):
        # Calculate the current learning rate
        progress = state.global_step / float(self.num_training_steps - 1)
        lr = self.start_lr * (self.end_lr / self.start_lr) ** progress
        
        # Set the learning rate for this step
        for param_group in kwargs["optimizer"].param_groups:
            param_group["lr"] = lr
        
        # Log the learning rate
        self.lrs.append(lr)

        # Check if there is any logged loss, and only then record it
        if state.log_history:
            if "loss" in state.log_history[-1]:
                # Record the loss from the latest log
                loss = state.log_history[-1]["loss"]
                self.losses.append(loss)
            else:
                # If for some reason a loss was not recorded, you might append a NaN or a previous loss
                # This part depends on how you want to handle missing loss values
                self.losses.append(float('nan'))  # or you might repeat the previous loss



    def on_train_end(self, args, state, control, **kwargs):
        # Before plotting, we should ensure that the number of learning rates 
        # and losses are the same. We'll trim the larger list.

        # Determine the size of the smallest list between lrs and losses
        min_size = min(len(self.lrs), len(self.losses))

        # Trim the lists to be of the same size
        self.lrs = self.lrs[:min_size]
        self.losses = self.losses[:min_size]

        # Now, you can plot without mismatch issues.
        plt.plot(self.lrs, self.losses)
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.show()
