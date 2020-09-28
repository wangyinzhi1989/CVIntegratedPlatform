import torch
import torch.nn as nn
import config
from utils.utils import *
from models.focal_loss import FocalLoss

class FreeLoss(nn.Module):
    def __init__(self, num_class):
        super(FreeLoss, self).__init__()
        self.num_class = num_class

    def forward(self, predictions, targets, anchors, anchor_t, gr):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        #构建目标
        tcls, tbox, indices, anchors = build_targets(predictions, targets, anchors, anchor_t)
        hyp = config.hyp

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['cls_pw']])).to(device)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['obj_pw']])).to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Losses
        nt = 0  # number of targets
        np = len(predictions)  # number of outputs
        balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        for i, pi in enumerate(predictions):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
                lbox += (1.0 - giou).mean()  # giou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - gr) + gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

                # Classification
                if self.num_class > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        s = 3 / np  # output count scaling
        lbox *= hyp['giou'] * s
        lobj *= hyp['obj'] * s * (1.4 if np == 4 else 1.)
        lcls *= hyp['cls'] * s
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()