from open_code import *

# ==================== 参数配置 ====================
class Config(Config):
    a_max=0.3
    topk = 2
    num_exps = 1
    margin=0.4
    log_dir = "plus"
    outputfile='plus.txt'
hparams = Config()


class DownstreamGeneral(DownstreamGeneral):
    def __init__(self, hparams,exp):
        super().__init__(hparams, exp)
        self.hp = hparams
        self.a_min = 0
        self.a_max = hparams.a_max
        self.margin=hparams.margin
        self.dataset = CustomEmoDataset(self.hp.datadir, self.hp.labelpath, maxseqlen=self.hp.maxseqlen)
        self.model = PretrainedRNNHead(n_classes=self.dataset.nemos, backend=self.hp.model_type)
        self.load_tapt_weights(self.hp.pretrained_path)
        counter = self.dataset.train_dataset.emos
        weights = torch.tensor([counter[c] for c in self.dataset.emoset]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        print(f"Weigh losses by prior distribution of each class: {weights}.")
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        if hasattr(self.dataset, 'val_dataset'):
            self.valid_met = ConfusionMetrics(self.dataset.nemos)
        if hasattr(self.dataset, 'test_dataset'):
            self.test_met = ConfusionMetrics(self.dataset.nemos)

    def similarity_loss(self, embeddings, labels, margin=0.4):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        B = labels.size(0)
        eye = torch.eye(B, device=labels.device, dtype=torch.bool)

        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & (~eye)
        negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0))

        # Positive similarity
        positive_sim = sim_matrix[positive_mask]
        if positive_sim.numel() > 0:
            positive_loss = (1.0 - positive_sim).mean()
        else:
            positive_loss = torch.tensor(0.0, device=embeddings.device)

        # Hard negative
        masked_negative = sim_matrix.masked_fill(~negative_mask, -1.0)
        hard_negative, _ = masked_negative.max(dim=1)
        hard_negative_loss = F.relu(hard_negative - margin).mean()
        sim_loss=2*positive_loss + hard_negative_loss
        # Total
        return sim_loss,positive_loss ,hard_negative_loss

    def training_step(self, batch, batch_idx):
        feats, length, label = batch
        pout, embedding = self(feats, length)
        ce_loss = self.criterion(pout, label)
        # 2. similarity loss
        sim_loss,positive_loss ,hard_negative_loss = self.similarity_loss(embedding,label,margin=self.margin)


        current_step = self.current_epoch * (len(self.train_dataloader())) + batch_idx
        total_steps = self.trainer.max_epochs * len(self.train_dataloader())
        progress = current_step / total_steps
        if progress<2/3:
            self.a = self.a_min + 0.5 * (self.a_max - self.a_min) * (1 - math.cos(math.pi * progress))
            loss = ce_loss + self.a*sim_loss
        else:
            loss=positive_loss+hard_negative_loss+1e-3*ce_loss
        self.log('ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('sim_loss', sim_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('positive_loss', positive_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('hard_negative_loss', hard_negative_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        feats, length, label = batch
        pout, embedding = self(feats, length)
        loss = self.criterion(pout, label)
        for l, p in zip(label, pout):
            self.valid_met.fit(int(l), int(p.argmax()))


    def test_step(self, batch, batch_idx):
        feats, label = batch
        length = torch.LongTensor([feats.size(1)]).to(label.device)
        pout, embedding = self(feats, length)
        self.test_met.fit(int(label), int(pout.argmax()))



def main():
    if not os.path.exists(hparams.saving_path):
        os.makedirs(hparams.saving_path)

    outputstr = f"a_max={hparams.a_max:.2f}, margin={hparams.margin:.2f}, topk={hparams.topk}epoch={hparams.max_epochs:.0f}\n"
    with open(hparams.outputfile, 'a') as f:
        f.write(outputstr)
    for exp in range(hparams.num_exps):
        exp+=3
        for ifold, foldlabel in enumerate(sorted(os.listdir(hparams.labeldir))):
            hparams.labelpath = os.path.join(hparams.labeldir, foldlabel)
            model = DownstreamGeneral(hparams,exp=exp)


            base_checkpoint = ModelCheckpoint(
                dirpath=hparams.saving_path+f"/sess{ifold+1}",
                filename=f'{hparams.log_dir}_{{valid_UAR:.4f}}',
                save_top_k=hparams.save_top_k,
                verbose=True,
                save_weights_only=True,
                monitor='valid_UAR',
                mode='max'
            )
            checkpoint_callback = DelayedModelCheckpoint(base_checkpoint, start_epoch=hparams.start_epoch)

            tb_logger = TensorBoardLogger(save_dir=hparams.log_dir, name="sess" + f"{ifold + 1}",
                                          version=f"exp={exp}_epoch_{hparams.max_epochs}")
            trainer = Trainer(
                precision=hparams.precision,
                amp_backend='native',
                gradient_clip_val=1.0,
                callbacks=[checkpoint_callback],
                max_epochs=hparams.max_epochs,
                check_val_every_n_epoch=1,
                gpus=1,
                logger=tb_logger,
                log_every_n_steps=10
            )

            trainer.fit(model)
            best_model_path = checkpoint_callback.best_model_path
            checkpoint = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            outputstr = f"+++sess:{ifold+1}+++\n"
            with open(hparams.outputfile, 'a') as f:
                f.write(outputstr)
            trainer.test(model)

if __name__ == '__main__':
    main()