def simple_accuracy(preds, labels):
    return (preds == labels).mean()
