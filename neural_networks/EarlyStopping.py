class EarlyStopping:
    def __init__(self, monitor='val_mae', patience=50, min_delta=0.00001, verbose=False):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_metric):
        score = -val_metric  # Because we want to maximize val_mae (lower is better)

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