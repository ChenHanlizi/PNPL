import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from utils import AverageMeter, label_transform
from geomloss import SamplesLoss
import numpy as np
import os
import cv2

def custom_alpha_cross_entropy(predict, soft_label, alpha):
    soft_label = soft_label.bool()
    softmax_p = F.softmax(predict, dim=1)
    sub = torch.masked_select(softmax_p, soft_label).view(predict.shape[0], -1)
    sub = sub[1:,:]
    # predict_class_prob = softmax_p.gather(1, label.view(-1, 1)).squeeze()
    diff = torch.abs(sub - alpha/(1+sub.shape[1]))
    loss = -torch.mean(diff)
    return loss
import random


def custom_cost(X,Y):
    if len(X.shape) == 2:
        N, D = X.shape
        M, D = Y.shape
        return (1-torch.eye(N,M)).cuda()
    if len(X.shape) == 3:
        B, N, D = X.shape
        B, M, D = Y.shape
        return torch.unsqueeze(1 - torch.eye(N, M), 0).repeat(B,1,1).cuda()
        
def train_clip(net, optimizer, scheduler, trainloader, run, epoch=None, proto=None, **options):
    losses = AverageMeter()
    loss_all = 0
    ori_to_modify, modify_to_ori = None, None
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        with torch.set_grad_enabled(True):
            output, _  = net(data)
            loss = F.cross_entropy(output, labels)
            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        losses.update(loss.item(), labels.size(0))
        run.log({'loss': loss.item()})
        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg)) 
        loss_all += losses.avg

    return loss_all

class Contrastiveloss(torch.nn.Module):
    def __init__(self, initial_temperature=0.5):
        super(Contrastiveloss, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, features, labels=None, mask=None):
       
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        temperature = torch.clamp(self.temperature, min=1e-6) 
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)     
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        
        num_positives_per_row  = torch.sum(positives_mask, axis=1)       
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdim=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdim=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        
        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        

        loss = -log_probs
        loss = loss.mean()

        return loss


def returnCAM(feature_conv, weight_softmax, class_idx):
    b, c, h, w = feature_conv.shape  
    output_cam = []
    for idx in class_idx: 
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h * w)))  
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化
        cam_img = np.uint8(255 * cam_img)  # 转为 uint8
        output_cam.append(cam_img)
    return output_cam

def train_nega_clip(net, optimizer, scheduler, trainloader, run, epoch=None,  proto=None, **options):
    losses = AverageMeter()
    loss_all = 0
    n_nega_ctx = options['NEGA_CTX']

    dataset = options['dataset']


    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        if options['POMP']:
            ori_to_modify, modify_to_ori = label_transform(labels.cpu().numpy(), options['POMP_k'], options['num_classes']-1)
            modified_labels = torch.tensor([ori_to_modify[label.item()] for label in labels]).cuda()
            labels = modified_labels
        else:
            ori_to_modify, modify_to_ori = None, None
             
        with torch.set_grad_enabled(True):
            dataset = options['dataset']
            output, text_features = net(data, modify_to_ori)
            visual_embedding = net.get_visual_features(data)
            # output.shape = [batch_size, nclass * 1+n_nega_ctx] [64,36]
            output_posi = output.view(-1, int(output.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)[:, :, 0]
            ensemble_text_features = text_features.view(int(text_features.shape[0]/(1+n_nega_ctx)), 1+n_nega_ctx, -1)
            positive_text_features = ensemble_text_features[:, 0, :]
            negative_text_features = ensemble_text_features[:, 1:, :] # (batch_size, n_nega_ctx , 512)
    
            loss_positive = F.cross_entropy(output_posi, labels)
            loss_prototype = 0
            if(options['prototype_weight'] != 0):
                loss_prototype = -torch.sum(torch.mul(positive_text_features, proto))
                
            loss_nega_to_other = 0
            loss_nega_to_posi = 0
            loss_nega_to_nega = 0
            # c_loss = 0
            con_loss=Contrastiveloss(initial_temperature=0.5)
            c_loss = con_loss(visual_embedding, labels, mask=None)
          
            if options['stage'] > 1:
                loss_positive *= 1e-8
                c_loss *= 1e-8
                # negative_features = negative_features.view(0)
                for i in range(negative_text_features.shape[0]):
                    negative_features = negative_text_features[i,:,:].float()
                    negative_features_mean = torch.mean(negative_features, dim=0, keepdim=True)
                    negative_features_mean_norm = negative_features_mean.norm(dim=-1, keepdim=True)

                    # Euclidean distance
                    # loss_nega_to_nega += -sum(torch.pdist(negative_features, p=2))

                    # Cosine distance
                    negative_features_norm = negative_features.norm(dim=-1, keepdim=True)
                    # nega_nega
                    # dot_product = negative_features_norm @ negative_features_norm.t()
                    # nega_mean
                    dot_product = negative_features_norm @ negative_features_mean_norm.t()
                    loss_nega_to_nega = loss_nega_to_nega -torch.mean(1-dot_product)
                loss_nega_to_nega /= negative_text_features.shape[0]   
                
                # print(output_negas.transpose(1,2))
                out_nega_forCE = output
                # print(labels)
                # print('outnegashape',out_nega_forCE.shape) [64,36]
                # create soft_target(1-hot) for negative samples and positive samples
                soft_target = torch.zeros(out_nega_forCE.shape).long().cuda()
                # print('softtarget shape',soft_target.shape) [64,36]
                idx = torch.arange(out_nega_forCE.shape[0]).cuda()
                soft_target.view(soft_target.shape[0], int(output.shape[1]/(1+n_nega_ctx)), -1)[idx, labels, :] = 1
                # print(soft_target)
                # print('softtarget shape',soft_target.shape) [64,36]
                
                # soft_target_np = out_nega_forCE.cpu().detach().numpy()
                # soft_target_file = 'soft_target.txt'
                # with open(soft_target_file, 'w') as file:
                #     for row in soft_target_np:
                #         file.write(" ".join(map(str, row)) + "\n")
                labels_nega = labels.reshape(1, -1).repeat(n_nega_ctx, 1).t().reshape(-1)
                # out_nega_forCE = torch.sigmoid(out_nega_forCE)
                # soft_target = soft_target.float()
                if options['open_set_method'] == 'MSP':
                    loss_fun = nn.MultiLabelSoftMarginLoss(reduction='mean')
                    loss_nega_to_other = loss_fun(out_nega_forCE, soft_target)
                else:
                    raise NotImplementedError
                all_class_dis = 0
                for i in range(negative_text_features.shape[0]):
                    positive_feature = positive_text_features[i:i+1,:].float()
                    negative_feature = negative_text_features[i,:,:].float()
                    positive_feature_norm = positive_feature/positive_feature.norm(dim=-1, keepdim=True)
                    negative_feature_norm = negative_feature/negative_feature.norm(dim=-1, keepdim=True)
                    dot_product = positive_feature_norm @ negative_feature_norm.t()
                    mean_cosine_dis = (1-dot_product).mean()
                    # all_class_dis += mean_cosine_dis
                    all_class_dis = all_class_dis+mean_cosine_dis
                    
                if options['open_set_method'] == 'MSP':
                    # loss_nega_to_posi += all_class_dis/negative_text_features.shape[0]
                    loss_nega_to_posi = loss_nega_to_posi+all_class_dis/negative_text_features.shape[0]
                else:
                    # loss_nega_to_posi += all_class_dis/negative_text_features.shape[0]
                    loss_nega_to_posi = loss_nega_to_posi+all_class_dis/negative_text_features.shape[0]
                
            # print('loss1',loss_nega_to_other)
            # print('loss2',loss_nega_to_posi)
            loss = c_loss + loss_positive + options['prototype_weight'] * loss_prototype \
                    + options['negative_weight']*loss_nega_to_other + options['distance_weight']*loss_nega_to_posi #+ 

            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        losses.update(loss.item(), labels.size(0))
        
        if (batch_idx+1) % options['print_freq'] == 0: 
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        # loss_all += losses.avg
        loss_all = loss_all+losses.avg
    run.log({'loss': loss_all}, step=epoch)
    return loss_all
# '''