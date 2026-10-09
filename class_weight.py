from open_code import *

# ==================== 参数配置 ====================
class Config(Config):
    topk = 2
    a_max = 0.3
    margin = 0.4
    num_exps = 1
    confusion_gamma = 0.3
    outputfile = 'class_w.txt'
    log_dir = "class_w"
    confusion_gamma = 0.3
hparams = Config()


def get_confusion_margin(self, labels):
    B = labels.size(0)
    margins = self.margin + self.confusion_gamma * self.current_confusion_ratio[labels]
    margins[torch.arange(B, device=labels.device), labels] = 0.0
    return margins

class DownstreamGeneral(DownstreamGeneral):
    def __init__(self, hparams, exp):
        super().__init__(hparams, exp)
        self.hp = hparams
        self.a_min = 0
        self.a_max = hparams.a_max
        self.margin = hparams.margin
        self.topk = hparams.topk
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
        self.feature_dim = 768
        self.n_classes = self.dataset.nemos
        self.confusion_gamma = hparams.confusion_gamma
        self.register_buffer('confusion_matrix', torch.zeros(self.n_classes, self.n_classes))
        self.register_buffer('current_confusion_ratio', torch.zeros(self.n_classes, self.n_classes))

    def get_classifier_weight(self):
        for module in reversed(self.model.linear_head):
            if isinstance(module, nn.Linear):
                return module.weight
        raise RuntimeError("Cannot find nn.Linear inside linear_head")

    def topk_confusion_boundary_loss(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        prototypes = F.normalize(self.get_classifier_weight(), p=2, dim=1)
        similarity = torch.matmul(embeddings, prototypes.T)
        B = embeddings.size(0)
        index = torch.arange(B, device=embeddings.device)
        positive_similarity = similarity[index, labels]
        negative_similarity = similarity.clone()
        negative_similarity[index, labels] = -1e4
        K = min(self.topk, self.n_classes - 1)
        topk_similarity, topk_index = negative_similarity.topk(K, dim=1)
        confusion_ratio = self.current_confusion_ratio.to(embeddings.device)
        topk_confusion = torch.gather(confusion_ratio[labels], 1, topk_index)
        dynamic_margin = self.margin + self.confusion_gamma * topk_confusion
        compact_loss = (1.0 - positive_similarity).mean()
        boundary_loss = F.relu(dynamic_margin + topk_similarity - positive_similarity.unsqueeze(1)).mean()
        total_loss = 2*compact_loss + boundary_loss
        return total_loss, compact_loss.detach(), boundary_loss.detach(), dynamic_margin.detach()

    def training_step(self, batch, batch_idx):
        feats, length, label = batch
        pout, embedding = self(feats, length)
        ce_loss = self.criterion(pout, label)
        proto_loss, compact_loss, boundary_loss, dynamic_margin = self.topk_confusion_boundary_loss(embedding, label)


        current_step = self.current_epoch * len(self.train_dataloader()) + batch_idx
        total_steps = self.trainer.max_epochs * len(self.train_dataloader())
        progress = current_step / total_steps
        if progress<2/3:
            self.a = self.a_min + 0.5 * (self.a_max - self.a_min) * (1 - math.cos(math.pi * progress))
            loss = ce_loss + self.a*proto_loss
        else:
            loss=compact_loss+1e-2*ce_loss
        self.log('ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('compact_loss', compact_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('boundary_loss', boundary_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        feats, length, label = batch
        pout, embedding = self(feats, length)
        prediction = pout.argmax(dim=1)
        with torch.no_grad():
            for y, p in zip(label, prediction):
                self.confusion_matrix[y.long(), p.long()] += 1
        for l, p in zip(label, prediction):
            self.valid_met.fit(int(l), int(p))

    def test_step(self, batch, batch_idx):
        feats, label = batch
        length = torch.LongTensor([feats.size(1)]).to(label.device)
        pout, embedding = self(feats, length)
        self.test_met.fit(int(label), int(pout.argmax()))

    def on_validation_epoch_end(self):
        cm = self.confusion_matrix.clone()
        row_sum = cm.sum(dim=1, keepdim=True).clamp_min(1.0)
        confusion_ratio = cm / row_sum
        confusion_ratio.fill_diagonal_(0.0)
        self.current_confusion_ratio = confusion_ratio.detach()
        self.confusion_matrix.zero_()
        self.log('valid_UAR', self.valid_met.uar,on_epoch=True,prog_bar=True)
        self.log('valid_precision', self.valid_met.precision.mean(),on_epoch=True,prog_bar=True)
        self.valid_met.clear()

def main():
    if not os.path.exists(hparams.saving_path):
        os.makedirs(hparams.saving_path)

    outputstr = f"a_max={hparams.a_max:.2f}, margin={hparams.margin:.2f}, topk={hparams.topk}epoch={hparams.max_epochs:.0f}\n"
    with open(hparams.outputfile, 'a') as f:
        f.write(outputstr)
    for exp in range(hparams.num_exps):
        for ifold, foldlabel in enumerate(sorted(os.listdir(hparams.labeldir))):
            hparams.labelpath = os.path.join(hparams.labeldir, foldlabel)
            model = DownstreamGeneral(hparams, exp=exp)
            base_checkpoint = ModelCheckpoint(
                dirpath=hparams.saving_path + f"/sess{ifold + 1}",
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
            best_model_path =checkpoint_callback.best_model_path
            checkpoint = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            outputstr = f"+++sess:{ifold+1}+++\n"
            with open(hparams.outputfile, 'a') as f:
                f.write(outputstr)
            trainer.test(model)

if __name__ == '__main__':
    main()