#%%

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import json
import soundfile as sf
import random
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import torch.optim as optim
from torch.utils import data
import pytorch_lightning.core.module as pl
import numpy as np
import torch.nn as nn
import torch
from transformers import Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,  # 【必须导入】
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

class SingleRandomBucketSampler(data.Sampler):
    def __init__(self, nbuckets, length, batch_size, drop_last, dynamic_batch):
        self.length = length
        self.batch_size = batch_size
        self.dynamic_batch = dynamic_batch
        self.drop_last = drop_last
        indices = np.argsort(length)
        split = len(indices) // nbuckets
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])

    def __iter__(self):
        if not self.dynamic_batch:
            random.shuffle(self.indices)
        for x in self.indices:
            random.shuffle(x)
        idxs = [i for x in self.indices for i in x]
        batches, batch, sum_len = [], [], 0
        for idx in idxs:
            batch.append(idx)
            sum_len += (self.length[idx] if self.dynamic_batch else 1)
            if sum_len >= self.batch_size:
                batches.append(batch)
                batch, sum_len = [], 0
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.dynamic_batch:
            return sum(self.length) // self.batch_size
        if self.drop_last:
            return len(self.length) // self.batch_size  # type: ignore
        else:
            return (len(self.length) + self.batch_size - 1) // self.batch_size
def RandomBucketSampler(nbuckets, length, batch_size, drop_last,dynamic_batch=True):
    if dynamic_batch:
        batch_size = int(16000 * float(batch_size) / 64. * 250)
        print (f"Using dynamic batching, the batch duration is {float(batch_size) / 16000} seconds")
    return SingleRandomBucketSampler(nbuckets, length, batch_size, drop_last, dynamic_batch)

class BaselineDataset(data.Dataset):
    def __init__(self, datadir, labelpath, maxseqlen):
        if not labelpath:
            self.datasetbase = [x for x in os.listdir(datadir) if x[-4:] == '.wav']
        else:
            with open(labelpath, 'r') as f:
                label = json.load(f)
            print (labelpath)
            self.datasetbase = list(label['Train'].keys())
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]
        self.maxseqlen = maxseqlen

        #Print statistics:
        l = len(self.dataset)
        print (f'Total {l} examples')

        self.lengths = []
        print ("Loading over the dataset once...")
        for dataname in tqdm(self.dataset):
            wav, _sr = sf.read(dataname)
            self.lengths.append(len(wav))
        avglen = float(sum(self.lengths)) / len(self.lengths) / 16000
        print (f"Average duration of audio: {avglen} sec")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        return wav.astype(np.float32)

    def seqCollate(self, batch):
        getlen = lambda x: x.shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)
        def trunc(x):
            if x.shape[0] >= target_seqlen:
                #RandomCrop
                start_point = random.randint(0, x.shape[0] - target_seqlen)
                x = x[start_point: start_point+target_seqlen]
                output_length = target_seqlen
            else:
                output_length = x.shape[0]
                over = target_seqlen - x.shape[0]
                x = np.pad(x, [0, over])
            ret = (x, output_length)
            return ret
        batch = list(map(trunc, batch))
        return default_collate(batch)
#%%
def prepare_mask(length, shape, dtype, device):
    #Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype, device=device
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[
        (torch.arange(mask.shape[0], device=device), length - 1)
    ] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask
class Wav2vec2PretrainWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2PT = Wav2Vec2ForPreTraining.from_pretrained("wav2vec2-base", revision='2dcc7b7f9b11f0ef271067e62599a27317a03114')
        self.wav2vec2 = self.wav2vec2PT.wav2vec2
        # self.wav2vec2PT.freeze_feature_extractor()

    def trainable_params(self):
        ret = list(self.wav2vec2PT.parameters())
        return ret
    def forward(self, x, length=None):
        # 必须处于训练模式才能计算预训练损失
        self.wav2vec2PT.train()

        batch_size, raw_sequence_length = x.shape
        sequence_length = self.get_feat_extract_output_lengths(raw_sequence_length)
        feat_shape = (batch_size, sequence_length)

        # 处理变长序列的 attention_mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length)
            attn_mask = prepare_mask(length, feat_shape, x.dtype, x.device)
        else:
            attn_mask = torch.ones(batch_size, sequence_length, device=x.device).bool()

        with torch.no_grad():
            # 1. 生成被掩码的时间步（True=被掩码）
            mask_time_indices = _compute_mask_indices(
                feat_shape,
                mask_prob=self.wav2vec2PT.config.mask_time_prob,      # 默认 0.065
                mask_length=self.wav2vec2PT.config.mask_time_length,  # 默认 10
                attention_mask=attn_mask,
                min_masks=2,
            )
            # 2. 【修复核心】采样负样本索引
            sampled_negative_indices = _sample_negative_indices(
                feat_shape,
                num_negatives=self.wav2vec2PT.config.num_negatives,  # 默认 100
                mask_time_indices=mask_time_indices,
            )
            sampled_negative_indices = torch.from_numpy(sampled_negative_indices).to(x.device)
            mask_time_indices = torch.from_numpy(mask_time_indices).to(x.device)

        outputs = self.wav2vec2PT(
            x,
            attention_mask=attn_mask,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
        )

        return outputs  # 包含 loss, contrastive_loss, diversity_loss

    #From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length
#%%

class ContinueFinetuningBaseline(pl.LightningModule):
    def __init__(self, maxstep, batch_size,
                 lr, warmup_step, maxseqlen,
                 datadir, labelpath,
                 use_bucket_sampler, train_bucket_size, val_bucket_size):
        super().__init__()
        self.maxstep = maxstep
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_step = warmup_step
        self.use_bucket_sampler = use_bucket_sampler
        self.train_bucket_size = train_bucket_size
        self.val_bucket_size = val_bucket_size
        self.wav2vec2 = Wav2vec2PretrainWrapper()
        self.maxseqlen = maxseqlen
        self.data = BaselineDataset(datadir, labelpath, maxseqlen=maxseqlen)
        numtraining = int(len(self.data) * 1.0)
        splits = [numtraining, len(self.data) - numtraining]
        self.traindata, self.valdata = data.random_split(self.data, splits, generator=torch.Generator().manual_seed(58))

    def train_dataloader(self):
        idxs = self.traindata.indices
        if self.use_bucket_sampler:
            length = [self.data.lengths[i] for i in idxs]
            length = [l if l < self.maxseqlen else self.maxseqlen for l in length]
        sampler = RandomBucketSampler(self.train_bucket_size, length, self.batch_size, drop_last=True)
        return data.DataLoader(self.traindata,
                               num_workers=4,
                               batch_sampler=sampler,
                               collate_fn=self.data.seqCollate)

    def configure_optimizers(self):
        parameters = self.wav2vec2.trainable_params()
        optimizer = optim.Adam(parameters, lr=self.lr)
        #Learning rate scheduler
        num_training_steps = self.maxstep
        num_warmup_steps = self.warmup_step
        num_flat_steps = int(0.05 * num_training_steps)
        def lambda_lr(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (num_warmup_steps + num_flat_steps)))
            )
        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lambda_lr),
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def forward(self, x, length):
        reps = self.wav2vec2(x, length)
        reps, pretrain_loss = reps.projected_states, reps.loss
        return pretrain_loss


    def training_step(self, batch, batch_idx):
        feats, length = batch
        loss = self(feats, length=length)
        return loss
def main():
    #%%
    saving_path="saving_file"
    datadir='Dataset/IEMOCAP/Audio_16k'
    last_mode=None#"TA_last"
    all_step=20000
    batch_size = 32
    lr=1e-4
    #%%
    checkpoint_callback = ModelCheckpoint(
        dirpath=saving_path,
        filename='TA_last',
        save_top_k=1,
        verbose=True,
        save_last=True
    )
    wrapper = Trainer(
        precision=16,
        amp_backend='native',
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=last_mode,
        max_steps=all_step,
        accelerator='gpu', devices=1

    )

    model = ContinueFinetuningBaseline(maxstep=all_step,
                                       batch_size=batch_size,
                                       lr=lr,
                                       warmup_step=100,
                                       maxseqlen=int(16000*10),
                                       datadir=datadir,
                                       labelpath=None,
                                       use_bucket_sampler=True,
                                       train_bucket_size=50,
                                       val_bucket_size=20
                                       )

    wrapper.fit(model)
    #%%
if __name__ == '__main__':
    main()