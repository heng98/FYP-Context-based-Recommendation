import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss for SPECTER

    L = max{(d(q, p) - d(q, n) + m), 0}
    """

    def __init__(self, distance: str = "cosine", margin: float = 1.0):
        super(TripletLoss, self).__init__()
        """
        Args:
            distance: the name of the distance function used: cosine | l2_norm
        """
        self.distance = distance
        if distance != "cosine" and distance != "l2_norm":
            raise NotImplementedError(
                f"Distance function [{distance}] is not recognize"
            )

        # self.margin = margin

    def forward(
        self, query: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: torch.Tensor
    ) -> torch.Tensor:
        if self.distance == "cosine":
            distance_positive = -F.cosine_similarity(query, positive)
            distance_negative = -F.cosine_similarity(query, negative)
        elif self.distance == "l2_norm":
            distance_positive = F.pairwise_distance(query, positive)
            distance_negative = F.pairwise_distance(query, negative)

        loss = F.relu(distance_positive - distance_negative + margin)
        loss = loss.mean()

        return loss