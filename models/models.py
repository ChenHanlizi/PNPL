import torch
import numpy as np
import time
import torch.nn as nn
from torch.nn import functional as F
from models.ABN import MultiBatchNorm
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
import re
from transformers import CLIPTokenizer
from transformers import CLIPModel
import open_clip

_tokenizer = _Tokenizer()

def returnCAM(feature_conv, weight_softmax, class_idx):
    b, c, h, w = feature_conv.shape  
    output_cam = []
    for idx in class_idx: 
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h * w)))  
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  
        cam_img = np.uint8(255 * cam_img)  
        output_cam.append(cam_img)
    return output_cam

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['N_CTX']
        ctx_init = cfg['CTX_INIT']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("\"{}\"", "")
            ctx_init = ctx_init.replace(".", "")
            ctx_init = ctx_init.replace("_", " ")
            words = re.findall(r'\b\w+\b', ctx_init)
            n_ctx = len(words)
            print('n_ctx', n_ctx)
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg['CSC']:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class OriginalCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.clip_model = clip_model
        self.cfg = cfg
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # template = self.cfg['CTX_INIT']
        # text_inputs = torch.cat([clip.tokenize(template.format(classname)) for classname in self.classnames]).cuda()
        # text_features = self.clip_model.encode_text(text_inputs)
        # image_features = self.clip_model.encode_image(image)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        # logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return logits, text_features
    
    
    
    
class NegaPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['N_CTX']
        ctx_init = cfg['CTX_INIT']
        if cfg['CSC']>0:
            ctx_init = None
        n_nega_ctx = cfg['NEGA_CTX']
        self.n_nega_ctx = n_nega_ctx
        self.csc = cfg['CSC']
        self.cfg = cfg
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("\"{}\"", "")
            ctx_init = ctx_init.replace(".", "")
            ctx_init = ctx_init.replace("_", " ")
            words = re.findall(r'\b\w+\b', ctx_init)
            n_ctx = len(words)
            prompt = clip.tokenize(ctx_init)
            prompt = prompt.cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) 
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            # print("prompt.shape", prompt.shape)
            # print(prompt)
            # print("embedding.shape", embedding.shape)
            ctx_vectors = ctx_vectors.view(1, ctx_vectors.shape[0], ctx_vectors.shape[1]) # class_posi, ctx, vector
            ctx_vectors = ctx_vectors.repeat(1+n_nega_ctx, 1, 1) 
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg['CSC']>0:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, 1+n_nega_ctx, n_ctx, ctx_dim, dtype=dtype).cuda()
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(1+n_nega_ctx, n_ctx, ctx_dim, dtype=dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        if ctx_vectors.dim() == 3:
            ctx_positive = ctx_vectors[0:1, :, :]
            ctx_negative = ctx_vectors[1:, :, :]
        else:
            ctx_positive = ctx_vectors[:, 0:1, :, :]
            ctx_negative = ctx_vectors[:, 1:, :, :]
        self.ctx_positive = nn.Parameter(ctx_positive)  # to be optimized
        if ctx_negative.shape[0] == 0:
            ctx_negative = torch.empty(0, dtype=dtype).cuda()
        # print('ctx_negative shape',ctx_negative.shape) # tensor[2,4,512]
        self.ctx_negative = nn.Parameter(ctx_negative)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        positive_prompts = [prompt_prefix + " " +  name   for name in classnames]
        negative_prompts = [prompt_prefix + " " + name  for name in classnames]
        # print('negative prompts', negative_prompts)
            
        positive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in positive_prompts]).cuda()
        # print(positive_tokenized_prompts[0])
        negative_tokenized_prompts = torch.cat([clip.tokenize(p) for p in negative_prompts]).cuda()
        # print('negative tokenized prompts', negative_tokenized_prompts)
        # tokenized_prompts:
        # tensor([ <start>    a     photo   of   a  positive [classname] . <end>
                # [49406,   320,  1125,   539,   320,  4844,  1929,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  2368,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  4558,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  6531,   269, 49407, 0 ...,0]])
        with torch.no_grad():
            positive_embedding = clip_model.token_embedding(positive_tokenized_prompts).type(dtype)
            negative_embedding = clip_model.token_embedding(negative_tokenized_prompts).type(dtype)
        
        positive_embedding = positive_embedding.view(positive_embedding.shape[0], 1, positive_embedding.shape[1], positive_embedding.shape[2])
        negative_embedding = negative_embedding.view(negative_embedding.shape[0], 1, negative_embedding.shape[1], negative_embedding.shape[2])
        negative_embedding = negative_embedding.repeat(1, n_nega_ctx, 1, 1)
        embedding = torch.cat([positive_embedding, negative_embedding], dim=1)
        positive_tokenized_prompts = positive_tokenized_prompts.view(positive_tokenized_prompts.shape[0], 1, positive_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.view(negative_tokenized_prompts.shape[0], 1, negative_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.repeat(1, n_nega_ctx, 1)
        tokenized_prompts = torch.cat([positive_tokenized_prompts, negative_tokenized_prompts], dim=1)
        tokenized_prompts = tokenized_prompts.view(tokenized_prompts.shape[0]*tokenized_prompts.shape[1], -1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, :, 1 + n_ctx :, :])  # positive prompt CLS, EOS
        if cfg['stage'] >= 2:
            self.register_buffer("positive_token_prefix", embedding[:, :1, :1, :])  # SOS
            self.register_buffer("positive_token_suffix", embedding[:, :1, 1 + n_ctx :, :])  # positive prompt CLS, EOS
            self.register_buffer("negative_token_prefix", embedding[:, 1:, :1, :])  # SOS
            self.register_buffer("negative_token_suffix", embedding[:, 1:, 1 + n_ctx :, :])
            self.positive_tokenized_prompts = positive_tokenized_prompts.view(positive_tokenized_prompts.shape[0]*positive_tokenized_prompts.shape[1], -1)
            # print('1',negative_tokenized_prompts.shape) [12,2,77]
            self.negative_tokenized_prompts = negative_tokenized_prompts.view(negative_tokenized_prompts.shape[0]*negative_tokenized_prompts.shape[1], -1)

        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self, modify_to_ori = None):
        # modify_to_ori is a dic that transform the modified labels to original ones
        ctx_positive = self.ctx_positive
        # print('ctx_positive', ctx_positive.shape)
        ctx_negative = self.ctx_negative
        # ctx_negative = ctx_negative[0:1, 0:1, :].repeat(ctx_negative.shape[0], ctx_negative.shape[1], 1)
        # make ctx_negative[0,0,:] to ctx_negative
        if ctx_negative.shape[0] == 0:
            if ctx_positive.dim() == 3:
                ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = ctx_positive
        else:   
            if ctx_positive.dim() == 3:
                diff = ctx_positive.shape[1] - ctx_negative.shape[1]
                additional_rows = torch.zeros((ctx_negative.shape[0], diff, ctx_negative.shape[2]) ).cuda()
                additional_rows = additional_rows.to(ctx_negative.dtype)
                ctx_negative = torch.cat([additional_rows, ctx_negative], dim=1)
                ctx = torch.cat([ctx_positive, ctx_negative], dim=0)
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = torch.cat([ctx_positive, ctx_negative], dim=1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if modify_to_ori is not None:
            ori_labels = list(modify_to_ori.values())
            ctx = ctx[ori_labels]
            prefix = prefix[ori_labels]
            suffix = suffix[ori_labels]
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,     # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim = 2,
        )
        
        return prompts
    def foward_positive(self):
        ctx_positive = self.ctx_positive
        if ctx_positive.dim() == 3:
            ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        else:
            ctx = ctx_positive
        prefix = self.positive_token_prefix
        suffix = self.positive_token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,     # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim = 2,
        )
        return prompts
    
    def foward_negative(self):
        ctx_negative = self.ctx_negative
        # print(ctx_negative.shape) #[2,4,512]
        if ctx_negative.dim() == 3:
            ctx = ctx_negative.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        else:
            ctx = ctx_negative
        prefix = self.negative_token_prefix
        suffix = self.negative_token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,     # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim = 2,
        )
        # print('forward negative prompts', prompts.shape) # [12,2,77,512]ucmp3
        return prompts
    def update_ctx_positive(self, ctx_posi):
        noise_range = 1e-5
        noise_dist = dist.Uniform(low=-noise_range, high=noise_range, )        
        if self.csc == 1:
            ctx_negative_repeated = ctx_posi.repeat(1, self.n_nega_ctx, 1, 1)
        else:
            ctx_negative_repeated = ctx_posi.repeat(self.n_nega_ctx, 1, 1)
        ctx_negative = ctx_negative_repeated + noise_dist.sample(ctx_negative_repeated.shape).to(self.ctx_negative.device)
        ctx_negative = ctx_negative.half()
        self.ctx_positive = nn.Parameter(ctx_posi, requires_grad=False)
        self.ctx_negative = nn.Parameter(ctx_negative, requires_grad=True)

    def update_ctx_negative(self, ctx_nega):
        self.ctx_negative = nn.Parameter(ctx_nega, requires_grad=False)
    def freeze_ctx_positive(self):
        self.ctx_positive = nn.Parameter(self.ctx_positive, requires_grad=False)
        
    def get_ctx_positive(self):
        return self.ctx_positive
    
class NegaTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.transformer.eval()
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        if(hasattr(clip_model, 'attn_mask')):
            self.attn_mask = clip_model.attn_mask
        else:
            self.attn_mask = None
        # print('attn_mask is ', self.attn_mask)
    
    def forward(self, prompts, tokenized_prompts):
        if len(prompts.shape) == 4:
            prompts = torch.flatten(prompts, start_dim=0, end_dim=1)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND (n_class*(1+n_neg)) * n_ctx * dim 
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(x.device)
            x = self.transformer(x, self.attn_mask)
        else:
            x = self.transformer(x)
        # x = self.transformer(x, self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # print("x shape: ", x.shape)
        # print("tokenized_prompts shape: ", tokenized_prompts.shape)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 确保 self.text_projection 和 x 的类型一致
        text_projection = self.text_projection.type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ text_projection

        return x

def cosine_similarity(tensor1, tensor2):
    # tensor1 和 tensor2 形状应该为 [N, 512]
    return (tensor1 * tensor2).sum(dim=1) / (tensor1.norm(dim=1) * tensor2.norm(dim=1))
from sklearn.preprocessing import StandardScaler
def visualize_features(image_features, positive_text_features, negative_text_features):
    # 创建一个StandardScaler对象
    scaler = StandardScaler()
    
    # 准备合并所有特征
    all_features = torch.cat([image_features, positive_text_features.squeeze(), negative_text_features.view(-1, 512)], dim=0)
    
    # 将特征转移到CPU，转换为numpy数组，然后应用标准化
    all_features = all_features.cpu().detach().numpy()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(all_features_scaled)

    # 获取各部分特征的数量
    num_images = image_features.shape[0]
    num_positives = positive_text_features.shape[0]
    num_negatives = negative_text_features.shape[0] * negative_text_features.shape[1]
    
    # 可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_features[:num_images, 0], reduced_features[:num_images, 1], c='red', marker='x', label='Image Features')
    plt.scatter(reduced_features[num_images:num_images+num_positives, 0], reduced_features[num_images:num_images+num_positives, 1], c='blue', marker='o', label='Positive Text Features')
    plt.scatter(reduced_features[num_images+num_positives:, 0], reduced_features[num_images+num_positives:, 1], c='green', marker='s', label='Negative Text Features')
    
    plt.legend()
    plt.title('Feature Visualization with Standardized Features')
    plt.savefig('4.png')


def integrate_and_reduce_features(image_features, positive_text_features, negative_text_features):
    # 合并所有特征为一个大的tensor
    all_features = torch.cat([
        image_features, 
        positive_text_features.squeeze(), 
        negative_text_features.view(-1, 512)
    ], dim=0).cpu().detach().numpy()  # 确保特征在CPU上并转换为NumPy数组

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(all_features)
    return reduced_features

class NegaPromptCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        delta = torch.zeros((3, 224, 224))
        delta.require_grad = True
        self.perturbation = torch.nn.Parameter(delta.float(), requires_grad=True)

        self.prompt_learner = NegaPromptLearner(cfg, classnames, clip_model).cuda()
        self.n_nega_ctx = cfg['NEGA_CTX']
        self.stage = cfg['stage']
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = NegaTextEncoder(clip_model).cuda()
        # self.text_encoder = TextEncoder(clip_model)
        # self.weight_yes = self.merge_yes_feature(classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.positive_text_features = None
        self.clip_model = clip_model
        self.cfg = cfg
    def forward_negative(self, image):
        # 将 noise 移动到 GPU，并调整到批次大小
        noise = self.perturbation.cuda().unsqueeze(0).expand(image.size(0), -1, -1, -1)
        
        image = image + noise

            
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # image_features.shape[batchsize, 512]
        negative_prompts = self.prompt_learner.foward_negative()
        # print(negative_prompts)
        negative_tokenized_prompts = self.prompt_learner.negative_tokenized_prompts
        negative_text_features = self.text_encoder(negative_prompts, negative_tokenized_prompts) #(1000*n_nega_ctx) * 512)
        # print(negative_text_features.shape) [12*2, 512] ucmp3 known12
        positive_text_features = self.positive_text_features # 1000*512
        #fusion the text_features that positive, negative, positive, negative, ...
        positive_text_features = positive_text_features.view(positive_text_features.shape[0], 1, -1)
        negative_text_features = negative_text_features.view(positive_text_features.shape[0], self.n_nega_ctx, -1)
        text_features = torch.cat([positive_text_features, negative_text_features], dim=1)
        # print(text_features.shape) # [12,3,512]
        text_features = text_features.view(text_features.shape[0]*text_features.shape[1], -1)
        # print(text_features.shape) [36, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print(text_features.shape) [36, 512]
        logit_scale = self.logit_scale.exp()
        image_features = image_features.type(self.dtype)
        text_features = text_features.type(self.dtype)
        logits = (logit_scale * image_features @ text_features.t())
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return logits, text_features
    
    def forward(self, image, modify_to_ori = None):
        if self.stage == 3:
            return self.forward_negative(image)
        # 将 noise 移动到 GPU，并调整到批次大小
        noise = self.perturbation.cuda().unsqueeze(0).expand(image.size(0), -1, -1, -1)
        

        image = image + noise


        prompts = self.prompt_learner(modify_to_ori)
        # prompt shape: [n_class, 1+n_neg, n_ctx, dim]
        tokenized_prompts = self.tokenized_prompts
        
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        image_features = image_features.type(self.dtype)
        text_features = text_features.type(self.dtype)
        logits = (logit_scale * image_features @ text_features.t())
        return logits, text_features
    
    def forward_test(self, image, text_features=None):
        # 将 noise 移动到 GPU，并调整到批次大小
        noise = self.perturbation.cuda().unsqueeze(0).expand(image.size(0), -1, -1, -1)
        
        image = image + noise

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        image_features = image_features.type(self.dtype)
        text_features = text_features.type(self.dtype)
        logits = (logit_scale * image_features @ text_features.t())
        # print(logits)
        return logits, text_features
    
    def get_perturbation(self, perturbation):
        self.perturbation = nn.Parameter(perturbation, requires_grad=False)

    def get_visual_features(self, image):
        # 将 noise 移动到 GPU，并调整到批次大小
        noise = self.perturbation.cuda().unsqueeze(0).expand(image.size(0), -1, -1, -1)
        image = image + noise


        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

            
    def get_ctx_posi(self, ctx_posi):
        self.prompt_learner.update_ctx_positive(ctx_posi)
        # get positive_text_features
        prompts = self.prompt_learner.foward_positive()
        tokenized_prompts = self.prompt_learner.positive_tokenized_prompts
        self.positive_text_features = self.text_encoder(prompts, tokenized_prompts)

    def get_ctx_nega(self, ctx_nega):
        self.prompt_learner.update_ctx_negative(ctx_nega)
    
    def freeze_ctx_posi(self):
        self.prompt_learner.freeze_ctx_positive()

    def radius(self):
        prompts = self.prompt_learner() 
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        n_nega_ctx = self.cfg['NEGA_CTX']
        ensemble_text_features = text_features.view(int(text_features.shape[0]/(1+n_nega_ctx)), 1+n_nega_ctx, -1)
        positive_text_features = ensemble_text_features[:, 0, :]
        negative_text_features = ensemble_text_features[:, 1:, :]
        radius = torch.Tensor(positive_text_features.shape[0], n_nega_ctx)
        logit_scale = self.logit_scale.exp()
        for i in range(positive_text_features.shape[0]):
            positive_feature = positive_text_features[i,:]
            negative_features = negative_text_features[i,:,:]
            
            cos_sim = torch.nn.functional.cosine_similarity(negative_features, positive_feature.unsqueeze(0), dim=1)
            one_radius = 1 - cos_sim
            
            # one_radius = logit_scale*positive_feature @ negative_features.t()
            
            radius[i, :] = one_radius
        
        
        return radius

    