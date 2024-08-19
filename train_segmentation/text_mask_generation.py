import cv2
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel



# 加载预训练模型
device = torch.device('cuda:0')
def TMG(batch_):

    class_names_20 = ['background', 'S', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'L56', 'L45', 'L34', 'L23', 'L12', 'T121', 'T1112', 'T1011', 'T910']
    model_path = '/root/data1/spine_my/spine/LViT/Test_session_12.19_15h52/models/best_model.pth'
    # model = my_novel(batch_size=16)
    # checkpoint = torch.load(model_path)  # 加载模型权重
    # model.load_state_dict(checkpoint)  # 将权重加载到模型中
    model = torch.load(model_path).to(device)

    # 模型架构
    tokenizer = model.tokenizer  # text-->token
    text_encoder =model.text_encoder  # token-->[b, len, 768]
    text_proj = model.text_proj # [b, len, 768]-->[b, 512]
    visual_encoder = model.visual_encoder
    vision_proj = model.vision_proj


    # 提取20个文本提示的 cls_token
    class_name_cls = []
    for class_name in class_names_20:
        text_token = tokenizer(class_name, return_tensors="pt").to(device)
        text_embeds = text_encoder(**text_token).last_hidden_state.to(device)
        text_embed_cls = F.normalize(text_proj(text_embeds[:,0,:]),dim=-1).squeeze(0)
        class_name_cls.append(text_embed_cls)
    class_name_cls = torch.stack(class_name_cls, dim=1).to(device)
    class_name_cls = class_name_cls.transpose(0, 1)
    class_names_cls_tokens = class_name_cls  # [20, 512]

    # 提取图像的 patch_tokens
    imgs = batch_.to(device)
    image_embeds = visual_encoder(imgs)  # [b, 513, 768]
    image_feat = F.normalize(vision_proj(image_embeds[:, 1:, :]), dim=-1)  # [b, 512, 512]
    image_patch_tokens = image_feat  # [b, 512, 512]

    # text_mask_generation
    text_masks_per_batch = []
    for img_index in range(len(image_patch_tokens)):
        temp_pred = np.zeros((len(class_names_cls_tokens), 256, 512))
        for text_index in range(len(class_names_cls_tokens)):
            attn_ai2at = image_patch_tokens[img_index] @ class_names_cls_tokens[text_index].unsqueeze(-1)  # (512, 1)<--[512, 512]@[512, 1]
            attn_ai2at = attn_ai2at.reshape(16, 32)  # [16, 32]
            attn_ai2at = F.interpolate(attn_ai2at.unsqueeze(0).unsqueeze(0), scale_factor=16, mode="nearest")[0][0]  # [256, 512]
            H, W = attn_ai2at.shape[-2:]  # [256, 512]
            attn_ai2at = attn_ai2at.reshape(H * W, 1)  # [256*512, 1]
            attn_ai2at = attn_ai2at.reshape(H, W)  # [256, 512]
            min_value, max_value = attn_ai2at.min(), attn_ai2at.max()
            norm_attn = (attn_ai2at - min_value) / (max_value - min_value)  # [256, 512]
            temp_pred[text_index] = norm_attn.cpu().detach().numpy()
        text_masks_per_batch.append(temp_pred)
    text_masks_per_batch = np.array(text_masks_per_batch)
    text_masks_per_batch = torch.from_numpy(text_masks_per_batch).to(device)

    return text_masks_per_batch
