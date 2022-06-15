import logging

class SimpleEarlyStopping:
    def __init__(self, patience, minimal_improvement, run_name):
        self.best_train_loss = 0
        self.steps_since_last_improvement = 0
        self.patience = patience
        self.minimal_improvement = minimal_improvement
        self.path = f"../models/{run_name}.pth"

    def early_stop(self, train_loss_value, model):
        if self.steps_since_last_improvement > self.patience:
            return True

        elif (self.best_train_loss == 0) or ((self.best_train_loss - train_loss_value) > self.minimal_improvement):
            self.steps_since_last_improvement = 0
            self.best_train_loss = train_loss_value

            logging.info(f"Saving the model")
            torch.save(model.state_dict(), self.path)

        else:
            self.steps_since_last_improvement += 1

        return False
