import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin

class MyDistributedDataParallel(LightningDistributedDataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        kwargs["batch_idx"] = inputs[1]
        kwargs = (kwargs, )
        inputs = ((inputs[0].to(torch.device('cuda:{}'.format(device_ids[0]))), ), )
        return inputs, kwargs


class MyDDP(DDPPlugin):

    def configure_ddp(self):
        #self.pre_configure_ddp()
        self.model = MyDistributedDataParallel(
            self.model,
            device_ids=self.determine_ddp_device_ids(),
            find_unused_parameters=True
        )
