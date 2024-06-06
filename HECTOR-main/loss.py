import torch
import torch.nn as nn


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)


class LabelSmoothing(nn.Module):
    """Loss function with label smoothing."""

    def __init__(self, level_sizes, padding_idx, criterion, smoothing=0.0):
        """
        :param level_sizes: number of labels at level i; eg. level_sizes[2] - number of labels at 2nd level
        :type level_sizes: list[int]
        :param padding_idx: id of a padding token
        :type padding_idx: int
        :param criterion: loss function (currently supported: kldiv, ce (CrossEntropy))
        :type criterion: str
        :param smoothing: smoothing value
        :type smoothing: bool
        """
        super(LabelSmoothing, self).__init__()
        if criterion == "kldiv":
            self.criterion = nn.KLDivLoss(reduction="sum")
        elif criterion == "ce":
            self.criterion = nn.CrossEntropyLoss(reduction="sum")
        else:
            raise Exception("Unknown criterion")

        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = sum(level_sizes)
        self.true_dist = None
        self.level_mask = self.make_level_mask(level_sizes)  # tgt_seq_len x tgt_vocab_size

    def forward(self, x, target):
        """
        :param x: model predictions
        :type x: 3d torch.tensor, shape (batch_size, tgt_seq_len, tgt_vocab_size)
        :param target: true labels
        :type target: 2d torch.tensor, shape (batch_size x tgt_seq_len)
        :return: float, loss value
        """
        batch_size = x.shape[0]

        x = x.contiguous().view(-1, x.size(-1))  # 3d --> 2d: (batch_size * tgt_seq_len) x tgt_vocab_size
        target = target.contiguous().view(-1)  # 2d --> 1d: (batch_size * tgt_seq_len)

        true_dist = self.generate_true_dist(x, target, batch_size)

        return self.criterion(x, true_dist.clone().detach())

    def generate_true_dist(self, x, target, batch_size):
        """Generate target distribution with label smoothing."""
        level_mask = self.level_mask.repeat(batch_size, 1).to(device)  # (batch_size * tgt_seq_len) x tgt_vocab_size

        true_dist = level_mask.data.clone()

        level_labels_count = torch.sum(level_mask, dim=1, keepdim=True)  # number of allowed labels at each position

        # distribute smoothing value throughout allowed labels at each position
        # distract 2 for true label and pad label
        smoothing = self.smoothing / (level_labels_count - 2)
        true_dist = true_dist * smoothing

        # set _confidence_ prob to the correct label
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # set zero prob to padding label
        true_dist[:, self.padding_idx] = 0

        # mask out padding labels in target
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)  # zero probs for padded positions

        return true_dist

    @staticmethod
    def make_level_mask(level_sizes):
        """
        For each position i = [0, n] in a tgt sequence, mask out labels of level [1, i - 1] and [i + 1, n]
        Only labels of level i or level 0 (special labels) can appear at position i
        :param level_sizes: number of labels at level i; eg. level_sizes[2] - number of labels at 2nd level
        :type level_sizes: list[int]
        :return: 2d torch.tensor, shape (nlevels, nlabels)
        """
        nspecial = level_sizes[0]  # number of labels from at level 0
        seq_len = len(level_sizes)
        nlabels = sum(level_sizes)

        mask = torch.zeros(seq_len, nlabels)

        for i in range(1, len(level_sizes)):
            start = sum(level_sizes[:i])  # number of labels from level 0 to i + 1
            end = sum(level_sizes[:i + 1])  # number of labels from level i + 1 to n

            mask[i - 1, :nspecial] = 1  # special labels are possible at every position
            mask[i - 1, start:end] = 1

        mask[seq_len - 1, :nspecial] = 1  # at the last position only special labels are possible (</s>
        return mask


class SimpleLossCompute:
    """Generate predictions and compute loss."""

    def __init__(self, criterion):
        """
        :param criterion: loss function
        """
        self.criterion = criterion

    def __call__(self, x, y, norm, valid=False):
        """
        :param x: decoder output
        :type x: torch.tensor, shape (batch_size, tgt_seq_len, tgt_vocab_size)
        :param y: target
        :type y: torch.tensor, shape (batch_size, tgt_seq_len)
        :param norm: total number of relevant labels in the batch (single integer)
        :type norm: int
        :return: total_loss (for the report), avg_loss (for backward step)
        """
        sloss = self.criterion(x, y) / norm
        return sloss.data * norm, sloss
