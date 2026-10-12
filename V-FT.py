from open_code import *


class Config(Config):
    # Training parameters
    batch_size = 64
    max_epochs = 16
    num_exps = 3
    log_dir = "tapt"
    outputfile = 'tapt.txt'


hparams = Config()


def main():
    if not os.path.exists(hparams.saving_path):
        os.makedirs(hparams.saving_path)

    for exp in range(hparams.num_exps):
        for ifold, foldlabel in enumerate(sorted(os.listdir(hparams.labeldir))):
            hparams.labelpath = os.path.join(hparams.labeldir, foldlabel)
            model = DownstreamGeneral(hparams, exp)
            base_checkpoint = ModelCheckpoint(
                dirpath=hparams.saving_path + f"/sess{ifold + 1}",
                filename=f'{hparams.log_dir}_{{valid_UAR:.4f}}',
                save_top_k=hparams.save_top_k,
                verbose=True,
                save_weights_only=True,
                monitor='valid_UAR',
                mode='max'
            )
            checkpoint_callback = DelayedModelCheckpoint(base_checkpoint, start_epoch=5)

            tb_logger = TensorBoardLogger(
                save_dir=hparams.log_dir,
                name="sess" + f"{ifold + 1}",
                version=f"exp={exp}_epoch_{hparams.max_epochs}"
            )
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
            outputstr = f"+++sess:{ifold + 1}+++\n"
            with open(hparams.outputfile, 'a') as f:
                f.write(outputstr)
            trainer.test(model)


if __name__ == '__main__':
    main()