import sys

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.maximum = 0
        self.minimum = sys.maxsize

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count
        self.maximum = max(self.maximum, val)
        self.minimum = min(self.minimum, val)

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count

    def get_maximum(self):
        return self.maximum

    def get_minimum(self):
        return self.minimum