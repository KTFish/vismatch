import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import py3_wget
from huggingface_hub import snapshot_download

from vismatch import THIRD_PARTY_DIR, BaseMatcher
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("CLIDD"))

from clidd import CLIDD  # noqa: E402
from model.model import Model  # noqa: E402


class CLIDDMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, score=-5, radius=2, *args, **kwargs):
        super().__init__(device, **kwargs)

        weights_path = self._get_weights()

        # CLIDD.__init__ hardcodes ./weights/{cfg}.pth
        self._net = CLIDD.__new__(CLIDD)
        nn.Module.__init__(self._net)
        self._net.top_k = max_num_keypoints
        self._net.radius = radius
        self._net.score_thresh = score

        self._net.model = Model(**CLIDD.cfgs["A48"])
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self._net.model.load_state_dict(state)
        self._net.model.eval().to(device)

        if radius > 0:
            self._net.mp = nn.MaxPool2d(radius * 2 + 1, 1, radius)

    @staticmethod
    def _get_weights() -> Path:
        cache_dir = Path(snapshot_download("vismatch/clidd"))
        weights_path = cache_dir / "A48.pth"
        if not weights_path.is_file():
            py3_wget.download_file(
                "https://github.com/HITCSC/CLIDD/releases/download/v1.0/A48.pth",
                weights_path,
            )
        return weights_path

    def preprocess(self, img):
        return img.unsqueeze(0)

    def _forward(self, img0, img1):
        out0 = self._net(self.preprocess(img0))[0]
        out1 = self._net(self.preprocess(img1))[0]

        kpts0, desc0 = out0["keypoints"], out0["descriptors"]
        kpts1, desc1 = out1["keypoints"], out1["descriptors"]

        if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return empty, empty, None, None, None, None, None

        idxs0, idxs1 = self._net.match(desc0, desc1)

        if len(idxs0) == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return empty, empty, None, None, None, None, None

        return (
            kpts0[idxs0].cpu().numpy(),
            kpts1[idxs1].cpu().numpy(),
            kpts0.cpu().numpy(),
            kpts1.cpu().numpy(),
            desc0.cpu().numpy(),
            desc1.cpu().numpy(),
            None,
        )
