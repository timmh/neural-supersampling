import torch
import torchvision.models


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.vgg.vgg16(pretrained=True)
        self.layer_to_name = {
            "3": "relu1_2",
            "8": "relu2_2",
            "15": "relu3_3",
            "22": "relu4_3"
        }
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, y, x):
        assert y.shape == x.shape
        B = y.shape[0]
        batch = torch.cat((y, x), dim=0)
        total_loss = 0
        for name, module in self.backbone.features._modules.items():
            batch = module(batch)
            if name in self.layer_to_name:
                total_loss += torch.mean((batch[:B] - batch[B:]) ** 2)
        return total_loss / len(self.layer_to_name)