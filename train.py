"""Paper simulation training: LeWM, residual, inverse and normalized recovery.

The loss and model construction preserve the settings used by the ICRA runs.
Only scalar losses are logged; planning diagnostics run after training.
"""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger
from omegaconf import open_dict, OmegaConf

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


INVERSE_INPUT_MODES = (
    "predicted_transition_concat",
    "true_transition_concat",
    "predicted_displacement",
)


def remove_optional_reporting_callbacks(trainer):
    """Disable reporting hooks auto-installed by stable-pretraining 0.1.6."""
    omitted = {
        "EnvironmentDumpCallback", "LogUnusedParametersOnce", "ModuleSummary", "SLURMInfo",
    }
    trainer.callbacks = [
        callback for callback in trainer.callbacks
        if not (type(callback).__module__.startswith("stable_pretraining.")
                and type(callback).__name__ in omitted)
    ]


def inverse_features(current, predicted, observed, mode):
    if mode == "predicted_transition_concat":
        return torch.cat((current, predicted), dim=-1)
    if mode == "true_transition_concat":
        return torch.cat((current, observed), dim=-1)
    if mode == "predicted_displacement":
        return predicted - current
    raise ValueError(f"Unknown inverse input mode: {mode}")


def normalized_recovery_loss(model, current, predicted, action_embedding, cfg):
    """Fixed unit-variance Gaussian NLL plus beta KL, as in the paper."""
    if not cfg.loss.mi.enabled:
        return predicted.new_zeros(())
    if cfg.loss.mi.grad_mode != "detach_action_encoder":
        raise ValueError("The paper MI objective uses detach_action_encoder")
    if not (cfg.loss.mi.target_normalize and cfg.loss.mi.unit_variance_prior):
        raise ValueError("The paper MI objective requires normalized targets and a unit prior")
    if cfg.loss.mi.learn_posterior_logvar:
        raise ValueError("The paper MI objective uses fixed unit posterior variance")

    target = action_embedding.detach()
    flat = target.reshape(-1, target.size(-1))
    mean = flat.mean(dim=0, keepdim=True).detach()
    std = flat.std(dim=0, unbiased=False, keepdim=True).detach().clamp_min(cfg.loss.mi.target_eps)
    shape = (1,) * (target.ndim - 1) + (target.size(-1),)
    target = (target - mean.view(shape)) / std.view(shape)
    mu = model.mi_posterior(torch.cat((current, predicted), dim=-1))
    # The constant term matches the original Gaussian NLL and is harmless for gradients.
    nll = 0.5 * ((target - mu).square() + torch.log(mu.new_tensor(2.0 * torch.pi))).sum(dim=-1).mean()
    kl = 0.5 * mu.square().sum(dim=-1).mean()
    return nll + cfg.loss.mi.kl_weight * kl


def compute_model_outputs(model, batch, cfg):
    """Expose the same transition tensors to the paper diagnostic scripts."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    history = cfg.wm.history_size
    num_preds = cfg.wm.num_preds
    output.update({
        "ctx_emb": output["emb"][:, :history],
        "ctx_act": output["act_emb"][:, :history],
        "tgt_emb": output["emb"][:, num_preds:num_preds + history],
    })
    output["pred_emb"] = model.predict(output["ctx_emb"], output["ctx_act"])
    return output


def compute_inverse_outputs(model, ctx_emb, pred_emb, tgt_emb, ctx_act,
                            tgt_act_emb, cfg, pred_loss):
    """Inverse metrics used by the factual rollout diagnostic."""
    if not cfg.loss.inverse.enabled:
        return {"inverse_loss": pred_loss.new_zeros(())}
    mode = cfg.loss.inverse.get("input_mode", "predicted_transition_concat")
    features = inverse_features(ctx_emb, pred_emb, tgt_emb, mode)
    target = tgt_act_emb.detach() if cfg.loss.inverse.grad_mode == "detach_action_encoder" else tgt_act_emb
    predicted_action = model.inverse(features)
    loss = F.mse_loss(predicted_action, target)
    return {
        "inverse_loss": loss,
        "inverse_cos": F.cosine_similarity(predicted_action, target, dim=-1).mean(),
        "inverse_to_pred_loss_ratio": loss / (pred_loss.detach() + 1.0e-8),
        "pred_act_emb": predicted_action,
        "tgt_act_emb_used": target,
    }


def paper_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    encoded = self.model.encode(batch)
    history = cfg.wm.history_size
    num_preds = cfg.wm.num_preds
    current = encoded["emb"][:, :history]
    observed = encoded["emb"][:, num_preds:num_preds + history]
    actions = encoded["act_emb"][:, :history]
    predicted = self.model.predict(current, actions)

    pred_loss = (predicted - observed).square().mean()
    sigreg_loss = self.sigreg(encoded["emb"].transpose(0, 1))
    inverse_loss = pred_loss.new_zeros(())
    if cfg.loss.inverse.enabled:
        mode = cfg.loss.inverse.input_mode
        features = inverse_features(current, predicted, observed, mode)
        inverse_target = actions.detach() if cfg.loss.inverse.grad_mode == "detach_action_encoder" else actions
        inverse_loss = F.mse_loss(self.model.inverse(features), inverse_target)
    mi_loss = normalized_recovery_loss(self.model, current, predicted, actions, cfg)
    total = (pred_loss + cfg.loss.sigreg.weight * sigreg_loss
             + cfg.loss.inverse.weight * inverse_loss + cfg.loss.mi.weight * mi_loss)

    values = {"loss": total, "pred_loss": pred_loss, "sigreg_loss": sigreg_loss,
              "inverse_loss": inverse_loss, "mi_loss": mi_loss}
    for name, value in values.items():
        self.log(f"{stage}/{name}", value.detach(), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch["pixels"].size(0))
    return values


def make_model(cfg):
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.embed_dim
    predictor = ARPredictor(
        num_frames=cfg.wm.history_size, input_dim=embed_dim,
        hidden_dim=hidden_dim, output_dim=hidden_dim, **cfg.predictor,
    )
    action_encoder = Embedder(
        input_dim=cfg.data.dataset.frameskip * cfg.wm.action_dim, emb_dim=embed_dim,
    )
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    inverse_head = nn.Identity()
    if cfg.loss.inverse.enabled:
        mode = cfg.loss.inverse.input_mode
        if mode not in INVERSE_INPUT_MODES:
            raise ValueError(f"Unknown inverse input mode: {mode}")
        inverse_head = MLP(input_dim=embed_dim if mode == "predicted_displacement" else 2 * embed_dim,
                           output_dim=embed_dim, hidden_dim=1024, norm_fn=nn.BatchNorm1d)
    mi_head = None
    if cfg.loss.mi.enabled:
        mi_head = MLP(input_dim=2 * embed_dim, output_dim=embed_dim,
                      hidden_dim=1024, norm_fn=nn.BatchNorm1d)
    return JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        inverse_head=inverse_head, mi_posterior_head=mi_head,
        projector=projector, pred_proj=pred_proj,
        residual_target=cfg.wm.residual_target,
    )


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    if cfg.repro.seed_everything:
        pl.seed_everything(cfg.seed, workers=True)
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if column.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, column, column))
            setattr(cfg.wm, f"{column}_dim", dataset.get_dim(column))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator)
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False)

    model = make_model(cfg)
    optimizers = {"model_opt": {
        "modules": "model", "optimizer": dict(cfg.optimizer),
        "scheduler": {"type": "LinearWarmupCosineAnnealingLR"}, "interval": "epoch",
    }}
    module = spt.Module(
        model=model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(paper_forward, cfg=cfg), optim=optimizers,
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)
    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    callbacks = [ModelObjectCallBack(run_dir, filename=cfg.output_model_name)]
    trainer = pl.Trainer(
        **cfg.trainer, callbacks=callbacks, num_sanity_val_steps=1,
        logger=[CSVLogger(save_dir=str(run_dir), name="csv_logs")],
        enable_checkpointing=True,
    )
    remove_optional_reporting_callbacks(trainer)
    manager = spt.Manager(
        trainer=trainer, module=module, data=data,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
