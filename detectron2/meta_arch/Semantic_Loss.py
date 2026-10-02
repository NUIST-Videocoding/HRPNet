import torch
import torch.nn as nn

class Semantic_Loss(nn.Module):
    def __init__(self):

        super(Semantic_Loss, self).__init__()
        print(f"[INFO] Semantic_Loss will be computed on p2, p3, p4, p5, p6.")

    def forward(self, generated_features, target_features):

        loss_p2 = nn.MSELoss()(generated_features["p2"], target_features["p2"])
        loss_p3 = nn.MSELoss()(generated_features["p3"], target_features["p3"])
        loss_p4 = nn.MSELoss()(generated_features["p4"], target_features["p4"])
        loss_p5 = nn.MSELoss()(generated_features["p5"], target_features["p5"])
        loss_p6 = nn.MSELoss()(generated_features["p6"], target_features["p6"])
        total_loss = loss_p2 + loss_p3 + loss_p4 + loss_p5 + loss_p6
        num_layers = 5
        average_loss = total_loss / num_layers
        
        return average_loss