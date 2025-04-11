import torch
from collections import defaultdict

class MetricLogger:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def add(self, name, value):
        """Add a new value to a metric."""
        self.metrics[name].append(value)
    
    def average(self, name):
        """Calculate the average of a metric."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return sum(self.metrics[name]) / len(self.metrics[name])
        else:
            return 0.0
    
    def global_average(self, name):
        """Calculate the average of a metric across all processes."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            local_avg = self.average(name)
            global_avg = torch.tensor(local_avg).cuda()
            torch.distributed.all_reduce(global_avg, op=torch.distributed.ReduceOp.SUM)
            global_avg /= torch.distributed.get_world_size()
            return global_avg.item()
        else:
            return 0.0
