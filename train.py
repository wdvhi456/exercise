import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from datasetsmodule.for1000 import FSDDataModule
from models.dif_unet import LitDimma
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/for1000.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)  # 全局随机种子
    if cfg.dataset.name == "for1000": #设置加载哪个数据集（目前只有一个）
        dm = FSDDataModule(config=cfg.dataset)
    Modle_Path='checkpoints/my_model/my_model-v2.ckpt'#断点加载
    model = LitDimma.load_from_checkpoint(Modle_Path)
    model=LitDimma(config=cfg)
    callbacks = [
        pl.callbacks.progress.TQDMProgressBar(),
        ModelCheckpoint(
            monitor="val/psnr",
            mode="max",
            save_top_k=cfg.logger.save_top_k,
            save_last=False,
            auto_insert_metric_name=False,
            filename=cfg.name,
            dirpath=f'{cfg.logger.checkpoint_dir}/{cfg.name}',

        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(

        accelerator=cfg.device,
        devices=1,
        callbacks=callbacks,
        # logger=logger, # uncomment to use wandb
        max_steps=cfg.iter,
        val_check_interval=cfg.eval_freq,
        #ckpt_path='checkpoints/my_model/my_model-v3.ckpt',

    )


    trainer.fit(model, dm)

    model = LitDimma.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, config=cfg
    )
    output = trainer.test(model, datamodule=dm)

#save results as json
    with open(f'{cfg.logger.checkpoint_dir}/{cfg.name}/results.json', 'w') as f:
         json.dump(output[0], f, indent=4)


