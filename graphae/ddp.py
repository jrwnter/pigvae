import torch
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin

class MyDistributedDataParallel(LightningDistributedDataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        kwargs["batch_idx"] = inputs[1]
        kwargs = (kwargs, )
        #inputs = ((inputs[0].to(torch.device('cuda:{}'.format(device_ids[0]))), ), )
        inputs = (([inputs[0][0].to(torch.device('cuda:{}'.format(device_ids[0]))),
                    inputs[0][1].to(torch.device('cuda:{}'.format(device_ids[0])))],),)
        return inputs, kwargs


class MyDDP(DDPPlugin):

    def configure_ddp(self, model, device_ids):
        model = MyDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        return model
