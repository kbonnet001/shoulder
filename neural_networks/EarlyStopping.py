class EarlyStopping:
    """ Early stopping mechanism to halt training when a monitored metric stops improving.

    Args:
        monitor (str): Metric to monitor for early stopping (e.g., 'val_mae'). Default is 'val_mae'.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 50.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement. Default is 0.00001.
        verbose (bool): If True, prints a message when early stopping is triggered. Default is False.
    """
    def __init__(self, monitor='val_mae', patience=50, min_delta=0.00001, verbose=False):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta 
        self.verbose = verbose
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_metric):
        """ Check if the training should be stopped early based on the validation metric.

        Args:
            val_metric (float): Current value of the monitored metric.
        
        Returns:
            bool: True if early stopping is triggered, otherwise False.
        """
        score = -val_metric # Assume lower is better; if higher is better, use score = val_metric
        # min_delta must always be smaller than val_metric. 
        # Ensure it is sufficiently small to avoid premature stopping!
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print("Early stopping")
                self.early_stop = True
        else:
            self.best_score = score
            self.epochs_no_improve = 0
            
    def __str__(self):
        """
        Provide a summary of the early stopping status.
        """
        status = "Enabled" if not self.early_stop else "Triggered"
        return f"EarlyStopping(monitor={self.monitor}, patience={self.patience}, min_delta={self.min_delta}, \
            verbose={self.verbose}, status={status})"
