import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.MYDist import MYDist
import clip

class MYARPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(MYARPLoss, self).__init__()
        self.use_gpu = options['use_gpu']
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.Dist = MYDist(num_classes=options['num_classes'], feat_dim=options['feat_dim'], classes_names=options['classes_names'])
        self.points = None
        
        num_classes = options['num_classes']
        classes_names=options['classes_names']
        self.unknown_p_v = nn.Parameter(torch.randn(num_classes, 3, 224, 224))
        self.unknown_p_t = nn.Parameter(0.1 * torch.randn(num_classes, 77, 512))

        self.p_t_tokenize = clip.tokenize(classes_names)
        
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)


    def forward(self, x, y, my_center, labels=None):
        my_center = my_center.float()
        dist_dot_p = self.Dist(x, center=my_center, metric='dot') 
        dist_l2_p = self.Dist(x, center=my_center) 
        logits = dist_l2_p - dist_dot_p 

        if labels is None: return logits, 0
        loss = F.cross_entropy(logits / self.temp, labels) 

        center_batch = my_center[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.margin_loss(self.radius, _dis_known, target) 

        loss = loss + self.weight_pl * loss_r

        return logits, loss

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss
