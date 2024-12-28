import wandb
import pytorch_lightning as pl
import torch
import torchmetrics
import einops
import lpips
from torchvision.utils import save_image
from  diffusers.models.unets import UNet2DModel
from torch.nn.functional import sigmoid, tanh
from my_selectors import get_loss,get_optimizers
import argparse
from omegaconf import OmegaConf

def normal_head(y, x):
    return torch.clip(y, 0, 1)

def dim_base_head(y, x):
    return torch.clip(y + x[:, :3, :, :], 0, 1)

def HE_base_head(y, x):
    return torch.clip(y + x[:, 3:6, :, :], 0, 1)

def sig_dim_base_head(y, x):
    return torch.clip(sigmoid(y) + x[:, :3, :, :], 0, 1)

def retinex_base_head(y, x):
    cm_improvement = tanh(y[:, :3, :, :])
    luminance_imp = sigmoid(y[:, 3, :, :])
    output = (x[:, 6:9, :, :] + cm_improvement) * (luminance_imp + x[:, 9, :, :])[:, None, :, :]
    return torch.clip(output, 0, 1)

def tanh_retinex_base_head(y, x):
    cm_improvement = tanh(y[:, :3, :, :])
    luminance_imp = tanh(y[:, 3, :, :])
    output = (x[:, 6:9, :, :] + cm_improvement) * (luminance_imp + x[:, 9, :, :])[:, None, :, :]
    return torch.clip(output, 0, 1)

HEADS = {
    "dim": dim_base_head,
    "HE": HE_base_head,
    "sig_dim": sig_dim_base_head,
    "retinex": retinex_base_head,
    "normal": normal_head,
}

class LitDimma(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.save_hyperparameters(config)

        # self.backbone = UNet2DModel(
        #     sample_size=100,
        #     in_channels=10,
        #     out_channels=(3 if self.config.model.head != "retinex" else 4),
        #     block_out_channels=self.config.model.channels,
        #     layers_per_block=self.config.model.layers_per_block,
        #     down_block_types=tuple(self.config.model.downblock for _ in range(len(self.config.model.channels))),
        #     up_block_types=tuple(self.config.model.upblock for _ in range(len(self.config.model.channels))),
        #     add_attention=self.config.model.add_attention,
        #     attention_head_dim=self.config.model.attention_head_dim,
        # )
        self.backbone = UNet2DModel(
            sample_size=400,  # the target image resolution
            in_channels=10,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
            add_attention=True,
            attention_head_dim=64,

        )
        self.head = HEADS[self.config.model.head]

        self.loss_fn = get_loss(config.loss)
        self.ssim_loss = lpips.LPIPS(net="vgg", verbose=False)

        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def forward(self, image, source_lightness, target_lightness):

        light_diff = (1000 * (source_lightness-target_lightness)).long()
        #light_diff = (1000 * (target_lightness - source_lightness)).long()
        output = self.backbone(image, timestep=light_diff).sample
        return self.head(output, image)



    def shared_step(self, batch):
        image, target = batch["image"], batch["target"]
        source_lightness = batch["source_lightness"]
        target_lightness = batch["target_lightness"]

        predicted = self.forward(image, source_lightness, target_lightness)

        loss = self.loss_fn(predicted, target).mean()
        if self.config.model.ssim_loss:
            loss += self.config.model.ssim_loss_weight * self.ssim_loss(2*predicted - 1, 2*target - 1).mean()

        return loss, predicted

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions = self.shared_step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_psnr(predictions, batch["target"])
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.val_ssim(predictions, batch["target"])
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        _, predictions = self.shared_step(batch)

        self.val_psnr(predictions, batch["target"])
        self.log("test/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.val_ssim(predictions, batch["target"])
        self.log("test/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)

        if self.logger:
            for i in range(min(self.config.model.save_images, predictions.size(0))):
                self.logger.experiment.log(
                    {
                        f"imgs/image_{i + (batch_idx*self.config.dataset.batch_size)}": [
                            wandb.Image(batch["target"][i]),
                            wandb.Image(predictions[i]),
                        ]
                    }
                )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction, _ = self(
            batch["image"], batch["source_lightness"], batch["target_lightness"]
        )
        return prediction

    def configure_optimizers(self):
        return get_optimizers(self, self.config.optimizer)

    def pre_img(self,img):
        output=self.backbone(img)
        return output


# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str, default="configs/for1000.yaml")
# args = parser.parse_args()
# cfg = OmegaConf.load(args.config)
# tensor = torch.zeros((1,10, 100, 100))
# difnn=LitDimma(cfg)
# y=difnn(tensor,123456789445,123456789445)
# print(tensor.shape)