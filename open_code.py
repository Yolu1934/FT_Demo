from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # 【新增】导入 TensorBoardLoggerssh -p 31646 root@connect.westb.seetacloud.com
from pytorch_lightning.callbacks import RichProgressBar
import os
import json
from collections import Counter
import soundfile as sf
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from torch.utils import data
from transformers import Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
import pytorch_lightning.core.lightning as pl
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn

from pytorch_lightning.callbacks import Callback


class DelayedModelCheckpoint(Callback):
    def __init__(self, checkpoint_callback, start_epoch=5):
        self.checkpoint_callback = checkpoint_callback
        self.start_epoch = start_epoch
        self._best_model_path = None

    @property
    def best_model_path(self):
        return self._best_model_path

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            self.checkpoint_callback.on_validation_end(trainer, pl_module)
            if hasattr(self.checkpoint_callback, 'best_model_path'):
                self._best_model_path = self.checkpoint_callback.best_model_path
        else:
            print(f"Epoch {trainer.current_epoch}: Skipping checkpoint save")
def prepare_mask(length, shape, dtype, device):
    mask = torch.zeros(shape, dtype=dtype, device=device)
    mask[(torch.arange(mask.shape[0], device=device), length - 1)] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask


class Wav2vec2Wrapper(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        local_model_path = r"wav2vec2-base"
        self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(
            local_model_path,
            local_files_only=True
        ).wav2vec2

        self.wav2vec2.encoder.config.gradient_checkpointing = False
        self.pretrain = pretrain

        # 下游任务使用 SpecAugment 配置
        self.mask_time_length = 15
        self.mask_time_prob = 0.08
        self.observe_time_prob = 0.0
        self.mask_feature_length = 64
        self.mask_feature_prob = 0.05

    def trainable_params(self):
        """返回可训练参数 - 下游任务只训练 encoder"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
        return list(self.wav2vec2.encoder.parameters())

    def forward(self, x, length=None):
        with torch.no_grad():
            # 特征提取
            x = self.wav2vec2.feature_extractor(x)
            x = x.transpose(1, 2)
            x, _ = self.wav2vec2.feature_projection(x)

            # 处理变长序列
            mask = None
            if length is not None:
                length = self.get_feat_extract_output_lengths(length)
                mask = prepare_mask(length, x.shape[:2], x.dtype, x.device)

            # SpecAugment（只在训练时）
            if self.training:
                batch_size, sequence_length, hidden_size = x.size()

                # 时间维度掩码
                if self.mask_time_prob > 0:
                    mask_time_indices = _compute_mask_indices(
                        (batch_size, sequence_length),
                        self.mask_time_prob,
                        self.mask_time_length,
                        min_masks=2
                    )
                    mask_time_indices = torch.as_tensor(mask_time_indices, device=x.device)
                    masked_indicies = mask_time_indices & mask if mask is not None else mask_time_indices
                    flip_mask = torch.rand((batch_size, sequence_length),
                                           device=masked_indicies.device) > self.observe_time_prob
                    x[masked_indicies & flip_mask] = self.wav2vec2.masked_spec_embed.to(x.dtype)

                # 特征维度掩码
                if self.mask_feature_prob > 0:
                    mask_feature_indices = _compute_mask_indices(
                        (batch_size, hidden_size),
                        self.mask_feature_prob,
                        self.mask_feature_length,
                        min_masks=1
                    )
                    if not torch.is_tensor(mask_feature_indices):
                        mask_feature_indices = torch.from_numpy(mask_feature_indices).to(x.device)
                    x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0

        # Transformer 编码
        x = self.wav2vec2.encoder(x, attention_mask=mask)[0]
        reps = F.relu(x)
        return reps

    def get_feat_extract_output_lengths(self, input_length):
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length


class ConfusionMetrics:
    """混淆矩阵评估指标"""

    def __init__(self, n_classes):
        self.nc = n_classes
        self.m = np.zeros([n_classes, n_classes], dtype=np.float32)

    def fit(self, target, pred):
        np.add.at(self.m, tuple([target, pred]), 1)

    def clear(self):
        self.m = np.zeros([self.nc, self.nc], dtype=np.float32)

    @property
    def recall(self):
        s = self.m.sum(1)
        s[s == 0] = np.inf
        return np.diag(self.m) / s

    @property
    def uar(self):
        return self.recall.mean()

    @property
    def war(self):
        return np.diag(self.m).sum() / self.m.sum()

    @property
    def macroF1(self):
        p = self.precision
        r = self.recall
        d = p + r
        d[d == 0] = np.inf
        return 2 * p * r / d

    @property
    def precision(self):
        s = self.m.sum(0)
        s[s == 0] = np.inf
        return np.diag(self.m) / s
    # 【新增】获取每个类别的精度（按类别分开）
    def get_per_class_precision(self):
        """返回每个类别的精度字典"""
        s = self.m.sum(0)  # 每列求和（预测为该类的总数）
        per_class_precision = np.diag(self.m) / s
        per_class_precision[s == 0] = 0  # 避免除以0，将分母为0的类别精度设为0
        return per_class_precision

    # 【新增】获取每个类别的召回率（按类别分开）
    def get_per_class_recall(self):
        """返回每个类别的召回率字典"""
        s = self.m.sum(1)  # 每行求和（实际为该类的总数）
        per_class_recall = np.diag(self.m) / s
        per_class_recall[s == 0] = 0  # 避免除以0
        return per_class_recall

    # 【新增】获取每个类别的F1（按类别分开）
    def get_per_class_f1(self):
        """返回每个类别的F1分数"""
        p = self.get_per_class_precision()
        r = self.get_per_class_recall()
        d = p + r
        d[d == 0] = 1  # 避免除以0
        return 2 * p * r / d


class _CustomEmoDataset(data.Dataset):
    def __init__(self, datadir, label, emoset, split, maxseqlen=12):
        super().__init__()
        self.maxseqlen = maxseqlen * 16000
        self.split = split
        self.label = label
        self.emos = Counter([self.label[n] for n in self.label.keys()])
        self.emoset = emoset
        self.labeldict = {k: i for i, k in enumerate(self.emoset)}
        self.datasetbase = list(self.label.keys())
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]

        print(f'Statistics of {self.split} splits:')
        for k, v in self.emos.items():
            print(f'{k}: {v} examples')
        print(f'Total {len(self.dataset)} examples')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        _label = self.label[self.datasetbase[i]]
        label = self.labeldict[_label]
        return wav.astype(np.float32), label



class CustomEmoDataset:
    def __init__(self, datadir, labeldir, maxseqlen):
        with open(labeldir, 'r') as f:
            self.label = json.load(f)
        self.emoset = sorted(list(set([emo for split in self.label.values() for emo in split.values()])))

        self.nemos = len(self.emoset)
        self.maxseqlen = maxseqlen * 16000

        self.train_dataset = _CustomEmoDataset(datadir, self.label['Train'], self.emoset, 'training')
        if self.label['Val']:
            self.val_dataset = _CustomEmoDataset(datadir, self.label['Val'], self.emoset, 'validation')
        if self.label['Test']:
            self.test_dataset = _CustomEmoDataset(datadir, self.label['Test'], self.emoset, 'testing')

    def seqCollate(self, batch):
        getlen = lambda x: x[0].shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)

        def trunc(x):
            x = list(x)
            if x[0].shape[0] >= target_seqlen:
                x[0] = x[0][:target_seqlen]
                output_length = target_seqlen
            else:
                output_length = x[0].shape[0]
                over = target_seqlen - x[0].shape[0]
                x[0] = np.pad(x[0], [0, over])
            return (x[0], output_length, x[1])

        batch = list(map(trunc, batch))
        return default_collate(batch)


class PretrainedRNNHead(pl.LightningModule):
    def __init__(self, n_classes, backend='wav2vec2'):
        super().__init__()
        self.backend = backend
        self.wav2vec2 = Wav2vec2Wrapper(pretrain=False)
        self.cosine_scale = 16.0
        feature_dim = 768
        self.fc = nn.Linear(768, n_classes, bias=False)
        # 余弦分类器（无偏置）
        self.linear_head = nn.Sequential(
            nn.ReLU(),
            self.fc
        )

    def trainable_params(self):
        return list(self.wav2vec2.trainable_params()) + list(self.linear_head.parameters())

    def forward(self, x, length):
        reps = self.wav2vec2(x, length)
        last_feat_pos = self.wav2vec2.get_feat_extract_output_lengths(length) - 1
        reps = reps.permute(1, 0, 2)

        masks = (torch.arange(reps.size(0), device=reps.device)
                 .expand(last_feat_pos.size(0), -1) < last_feat_pos.unsqueeze(1))
        masks = masks.float()

        embedding = (reps * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)

        # 余弦分类器
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        weight_norm = F.normalize(self.fc.weight, p=2, dim=1)  # 直接访问 self.fc
        logits = self.cosine_scale * torch.matmul(embedding_norm, weight_norm.T)

        return logits, embedding
class DownstreamGeneral(LightningModule):
    def __init__(self, hparams,exp):
        super().__init__()
        self.hp = hparams
        self.dataset = CustomEmoDataset(self.hp.datadir, self.hp.labelpath, maxseqlen=self.hp.maxseqlen)
        self.model = PretrainedRNNHead(n_classes=self.dataset.nemos, backend=self.hp.model_type)
        if self.hp.pretrained_path is not None:
            self.load_tapt_weights(self.hp.pretrained_path)
        # 类别权重（处理不平衡）
        counter = self.dataset.train_dataset.emos
        weights = torch.tensor([counter[c] for c in self.dataset.emoset]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        print(f"Weigh losses by prior distribution of each class: {weights}.")
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        # 评估指标
        if hasattr(self.dataset, 'val_dataset'):
            self.valid_met = ConfusionMetrics(self.dataset.nemos)
        if hasattr(self.dataset, 'test_dataset'):
            self.test_met = ConfusionMetrics(self.dataset.nemos)

    def load_tapt_weights(self, pretrained_path):
        # print(f"\n{'=' * 50}")
        # print(f"Loading TAPT pretrained weights from: {pretrained_path}")
        # print(f"{'=' * 50}\n")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        new_state_dict = {}
        old_prefix = "wav2vec2.wav2vec2PT.wav2vec2."

        for key, value in state_dict.items():
            if key.startswith(old_prefix):
                new_key = key[len(old_prefix):]
                new_state_dict[new_key] = value

        missing, unexpected = self.model.wav2vec2.wav2vec2.load_state_dict(new_state_dict, strict=True)


    def forward(self, x, length):
        return self.model(x, length)
    def configure_optimizers(self):
        return optim.Adam(self.model.trainable_params(), lr=self.hp.lr)
    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset.train_dataset,
            collate_fn=self.dataset.seqCollate,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=self.hp.nworkers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset.val_dataset,
            collate_fn=self.dataset.seqCollate,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=self.hp.nworkers,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset.test_dataset,
            batch_size=1,
            num_workers=self.hp.nworkers,
            drop_last=False
        )

    def training_step(self, batch, batch_idx):
        feats, length, label = batch
        pout, embedding = self(feats, length)
        loss= self.criterion(pout, label)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feats, length, label = batch
        pout, embedding = self(feats, length)
        loss= self.criterion(pout, label)
        for l, p in zip(label, pout):
            self.valid_met.fit(int(l), int(p.argmax()))

    def on_validation_epoch_end(self):
        self.log('valid_UAR', self.valid_met.uar,on_epoch=True,prog_bar=True)
        self.log('valid_precision', self.valid_met.precision.mean(),on_epoch=True,prog_bar=True)
        self.valid_met.clear()

    def test_step(self, batch, batch_idx):
        feats, label = batch
        length = torch.LongTensor([feats.size(1)]).to(label.device)
        pout, embedding = self(feats, length)
        loss= self.criterion(pout, label)
        self.test_met.fit(int(label), int(pout.argmax()))

    def on_test_epoch_end(self):
        # 原有的整体指标
        outputstr = f"UAR [%]={self.test_met.uar * 100:.4f}, Precision[%]={self.test_met.precision.mean() * 100:.4f}\n"
        self.log('test_Uar', self.test_met.uar)
        self.log('test_Precision', self.test_met.precision.mean())
        precision_array = self.test_met.precision
        for i, emo_name in enumerate(self.dataset.emoset):
            self.log(f'test_precision_{emo_name}', precision_array[i])
            outputstr += f"precision{emo_name}: {precision_array[i]}\n"
        with open(self.hp.outputfile, 'a') as f:
            f.write(outputstr)



        self.test_met.clear()

class Config:
    batch_size = 64
    start_epoch = 6
    max_epochs = 16
    num_exps = 3


    lr = 1e-4
    maxseqlen = 10
    nworkers = 8
    precision = 16
    save_top_k = 1
    datadir = r"Dataset/IEMOCAP/Audio_16k"
    labeldir = r"Dataset/IEMOCAP/labels_sess/"
    saving_path = "best_model"
    pretrained_path ="saving_file/last.ckpt"
    model_type = 'wav2vec2'
