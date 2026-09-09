from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import sys
from unittest.mock import MagicMock

try:
    import torch_scatter
except Exception:
    sys.modules["torch_scatter"] = MagicMock()

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR / "UniCorrn")

from unicorrn.model import build_model
from unicorrn.utils import safe_load_weights
from unicorrn.utils.config import read_yaml_config
from unicorrn.inference import init_query_points, coarse_to_fine


class UnicorrnMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, grid_size=4, weights_path=None, *args, **kwargs):
        super().__init__(device, **kwargs)
        if not torch.cuda.is_available():
            raise RuntimeError("UniCorrn matcher requires GPU")
        self.grid_size = grid_size
        self.max_num_keypoints = max_num_keypoints

        if weights_path is None:
            weights_path = hf_hub_download(repo_id="prajnan/unicorrn", filename="UniCorrn_Large_Stage2.pth")

        weights_path = Path(weights_path)
        config_path = THIRD_PARTY_DIR / "UniCorrn" / "configs" / "models" / "unicorrn_large_stage2.yml"

        model_cfg = read_yaml_config(config_path)
        self.model = build_model(model_cfg.NAME, cfg=model_cfg)
        weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        safe_load_weights(self.model, weights["model"])
        self.model = self.model.to(self.device).eval()

    def _forward(self, img0, img1):
        img0_np = (img0.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img1_np = (img1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        H, W = img0_np.shape[:2]
        queries = init_query_points(H, W, grid_size=self.grid_size).view(-1, 2).numpy()

        kpts0, kpts1, confidence, _ = coarse_to_fine(
            img0_np,
            img1_np,
            queries,
            self.model,
            unified_model=True,
        )

        if confidence is not None:
            confidence = confidence.squeeze()
            if isinstance(confidence, torch.Tensor):
                confidence = confidence.cpu().numpy()

        if isinstance(kpts0, torch.Tensor):
            kpts0 = kpts0.cpu().numpy()
        if isinstance(kpts1, torch.Tensor):
            kpts1 = kpts1.cpu().numpy()

        if len(kpts0) == 0:
            return (
                np.empty((0, 2)),
                np.empty((0, 2)),
                None,
                None,
                None,
                None,
                np.empty((0,)),
            )

        return (
            kpts0,
            kpts1,
            None,
            None,
            None,
            None,
            confidence,
        )
