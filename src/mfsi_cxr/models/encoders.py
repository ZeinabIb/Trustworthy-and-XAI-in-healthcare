"""Image and text encoders (Phase 2.1). Interface: per-instance features, (N, feature_dim).

Image: TorchXRayVision DenseNet121 `densenet121-res224-mimic_ch` (MIMIC-CXR only; a CheXpert-trained
checkpoint would have seen CheXpert Plus test patients). The weight file's sha256 is checked. Features are
the DenseNet trunk + ReLU + global average pool (1024-d), as XRV's own `features2`.

Text: CXR-BERT-specialized at a pinned commit, loaded as a plain `BertModel` (no remote code; its projection
and masked-LM heads are not used). Feature = [CLS] of the last layer (768-d).

`model.weights: tiny_random` (smoke/tests) swaps in small random encoders with the same interface and a
hashing word tokenizer, so nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

WEIGHT_MODES = ("pretrained", "tiny_random")


class CheckpointError(RuntimeError):
    """A pretrained checkpoint does not match its pinned version or hash."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_mode(cfg: dict[str, Any]) -> str:
    mode = cfg["model"]["weights"]
    if mode not in WEIGHT_MODES:
        raise ValueError(f"model.weights must be one of {WEIGHT_MODES}, got {mode!r}")
    return mode


class ImageEncoder(nn.Module):
    """(N, C, H, W) -> (N, feature_dim)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        icfg = cfg["model"]["image_encoder"]
        self.feature_dim = icfg["feature_dim"]
        self.unfreeze_names = list(icfg["unfreeze"])
        if _weights_mode(cfg) == "pretrained":
            self.trunk = self._load_xrv(icfg)
        else:
            c = cfg["model"]["tiny"]["image_channels"]
            self.trunk = nn.Sequential(
                nn.Conv2d(cfg["image"]["channels"], c, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(c, self.feature_dim, 3, stride=2, padding=1),
            )
            self.unfreeze_names = []  # tiny trunk is trained in full

    @staticmethod
    def _load_xrv(icfg: dict[str, Any]) -> nn.Module:
        import torchxrayvision as xrv
        from torchxrayvision.models import model_urls

        if xrv.__version__ != icfg["xrv_version"]:
            raise CheckpointError(f"torchxrayvision {xrv.__version__} != pinned {icfg['xrv_version']}")
        model = xrv.models.DenseNet(weights=icfg["xrv_weights"])
        url = model_urls[icfg["xrv_weights"]]["weights_url"]
        path = Path.home() / ".torchxrayvision" / "models_data" / Path(url).name
        digest = _sha256(path)
        if digest != icfg["sha256"]:
            raise CheckpointError(f"{path.name}: sha256 {digest} != pinned {icfg['sha256']}")
        return model.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        return nn.functional.adaptive_avg_pool2d(nn.functional.relu(features), 1).flatten(1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Warm-up: everything frozen. After it: only the configured last blocks (all of a tiny trunk)."""
        for name, p in self.trunk.named_parameters():
            top = name.split(".")[0]
            p.requires_grad = trainable and (not self.unfreeze_names or top in self.unfreeze_names)


class HashTokenizer:
    """Word-level hashing tokenizer for tiny_random (no vocabulary file). Same call shape as HF tokenizers."""

    PAD, CLS, SEP = 0, 1, 2
    _word = re.compile(r"\w+|[^\w\s]")

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab

    def token_ids(self, text: str) -> list[int]:
        return [3 + zlib.crc32(w.lower().encode()) % (self.vocab - 3) for w in self._word.findall(text)]

    def __call__(self, texts: list[str], max_length: int, **_: Any) -> dict[str, torch.Tensor]:
        rows = [[self.CLS, *self.token_ids(t)[: max_length - 2], self.SEP] for t in texts]
        n = max((len(r) for r in rows), default=2)
        ids = torch.full((len(rows), n), self.PAD, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class TextEncoder(nn.Module):
    """list of N documents -> (N, feature_dim) [CLS] features."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        from transformers import BertConfig, BertModel

        tcfg = cfg["model"]["text_encoder"]
        self.max_len = {"index": tcfg["max_len_index"], "prior": tcfg["max_len_prior"]}
        self.unfreeze_layers = tcfg["unfreeze_layers"]
        if _weights_mode(cfg) == "pretrained":
            from transformers import BertTokenizerFast

            repo, rev = tcfg["repo"], tcfg["revision"]
            config = BertConfig.from_pretrained(repo, revision=rev)
            self.bert = BertModel.from_pretrained(repo, revision=rev, config=config, add_pooling_layer=False,
                                                  use_safetensors=True)
            self.tokenizer = BertTokenizerFast.from_pretrained(repo, revision=rev)
            if self.bert.config.hidden_size != tcfg["feature_dim"]:
                raise CheckpointError(f"text hidden size {self.bert.config.hidden_size} != feature_dim")
        else:
            tiny = cfg["model"]["tiny"]
            config = BertConfig(vocab_size=tiny["text_vocab"], hidden_size=tiny["text_hidden"],
                                num_hidden_layers=2, num_attention_heads=2,
                                intermediate_size=2 * tiny["text_hidden"],
                                max_position_embeddings=max(self.max_len.values()))
            self.bert = BertModel(config, add_pooling_layer=False)
            self.tokenizer = HashTokenizer(tiny["text_vocab"])
            self.unfreeze_layers = config.num_hidden_layers
        self.feature_dim = self.bert.config.hidden_size

    def tokenize(self, texts: list[str], kind: str) -> dict[str, torch.Tensor]:
        """kind: 'index' (pre-read text) or 'prior' (a prior report). Over-long documents are truncated."""
        return self.tokenizer(texts, max_length=self.max_len[kind], truncation=True, padding=True,
                              return_tensors="pt")

    def forward(self, texts: list[str], kind: str) -> torch.Tensor:
        device = next(self.bert.parameters()).device
        if not texts:
            return torch.zeros(0, self.feature_dim, device=device)
        enc = {k: v.to(device) for k, v in self.tokenize(texts, kind).items()}
        return self.bert(**enc).last_hidden_state[:, 0]

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Warm-up: frozen. After it: the last `unfreeze_layers` encoder layers only."""
        layers = self.bert.encoder.layer
        first = len(layers) - self.unfreeze_layers
        for p in self.bert.parameters():
            p.requires_grad = False
        if trainable:
            for layer in layers[first:]:
                for p in layer.parameters():
                    p.requires_grad = True
