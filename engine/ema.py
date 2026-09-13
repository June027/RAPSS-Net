import torch
import copy


class ModelEMA:
    """模型指数滑动平均抽象类"""

    def __init__(self, model, decay=0.999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def update(self, model):
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model_sd = raw_model.state_dict()
        with torch.no_grad():
            for name, ema_v in self.module.state_dict().items():
                if name not in raw_model_sd:
                    continue
                model_v = raw_model_sd[name]
                if self.device is not None:
                    model_v = model_v.detach().to(device=self.device, non_blocking=True)

                # 🌟 修复 3：判断是否为浮点数。整数类型（如 num_batches_tracked）直接硬拷贝
                # 彻底杜绝 RuntimeError: result type Float can't be cast to the desired output type Long
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)
                else:
                    ema_v.copy_(model_v)
