import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def t2t_contrastive(text_embeddings_learn: torch.Tensor,
                    text_embeddings_fix: torch.Tensor,
                    temperature=0.07,
                    reduction='mean'):
    """

    Args:
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
    """
    # Calculating the Contrastive Loss
    assert text_embeddings_learn.dim() == 2 and text_embeddings_fix.dim() == 2
    assert text_embeddings_learn.shape[0] == text_embeddings_fix.shape[0]
    assert text_embeddings_learn.shape[1] == text_embeddings_fix.shape[1]

    num_classes = text_embeddings_learn.shape[0]
    similarity = (text_embeddings_learn @ text_embeddings_fix.T) / temperature
    loss_1 = F.cross_entropy(similarity, torch.arange(num_classes).to(similarity.device), reduction=reduction)
    loss_2 = F.cross_entropy(similarity.T, torch.arange(num_classes).to(similarity.device), reduction=reduction)
    loss = (loss_1 + loss_2) / 2.0

    # loss = F.l1_loss(text_embeddings_learn @ text_embeddings_learn.T, text_embeddings_fix @ text_embeddings_fix.T, reduction=reduction)

    return loss


@LOSSES.register_module()
class Text2TextContrastiveLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_t2t'.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_t2t',
                 temperature=0.07,
                 text_embeddings=None):
        super(Text2TextContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.text_embeddings_fix = torch.load(text_embeddings).requires_grad_(False)
        self.text_embeddings_fix_normalize = F.normalize(self.text_embeddings_fix, p=2, dim=-1)
        self.criterion = t2t_contrastive
        self._loss_name = loss_name

    def forward(self,
                text_embeddings_learn_normalize,
                _,
                reduction_override=None,
                weight=None,
                ignore_index=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * self.criterion(
            text_embeddings_learn_normalize,
            self.text_embeddings_fix_normalize.to(text_embeddings_learn_normalize.device),
            temperature=self.temperature,
            reduction=reduction,
            **kwargs)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
