import torch
import torch.nn as nn

class DinoV2FeatureExtractor(nn.Module):

    def __init__(self, model_name='dinov3_vits16plus', device=None):
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        local_repo_path = '/data/fmy/hub_repos/dinov3'
        self.dinov2 = torch.hub.load(local_repo_path, model_name, source='local', weights='/data/fmy/hub_repos/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
        self.dinov2.to(self.device)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        

    def forward(self, img_tensor):

        with torch.no_grad():
            features = self.dinov2.get_intermediate_layers(
                img_tensor, 
                n=[2, 5, 8, 11],  
                reshape=True
            )
        
        return features