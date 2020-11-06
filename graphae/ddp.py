import torch
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel


class MyDistributedDataParallel(LightningDistributedDataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        kwargs["batch_idx"] = inputs[1]
        kwargs = (kwargs, )
        inputs = ((inputs[0].to(torch.device('cuda:{}'.format(device_ids[0]))), ), )
        return inputs, kwargs
