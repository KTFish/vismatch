import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from vismatch import THIRD_PARTY_DIR, BaseMatcher
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("CLIDD"))

from clidd import CLIDD
from model.model import Model


class CLIDDMatcher(BaseMatcher):
    """CLIDD keypoints and descriptors, matched with its own mutual nearest neighbour matcher, as in CLIDD's demo."""

    def __init__(self, device="cpu", max_num_keypoints=2048, variant="A48", score=-5, radius=2, *args, **kwargs):
        super().__init__(device, **kwargs)
        assert "cuda" in self.device, "CLIDD runs its triton kernels on cuda only"
        variant = variant.upper()
        assert variant in CLIDD.cfgs, f"Unknown CLIDD variant {variant}. Choose from {list(CLIDD.cfgs)}"

        # CLIDD.__init__ loads ./weights/{cfg}.pth from the working directory, so build it by hand
        self._net = CLIDD.__new__(CLIDD)
        nn.Module.__init__(self._net)
        self._net.top_k = max_num_keypoints
        self._net.radius = radius
        self._net.score_thresh = score
        self._net.model = Model(**CLIDD.cfgs[variant])
        weights = torch.load(hf_hub_download("vismatch/clidd", f"{variant}.pth"), map_location="cpu", weights_only=True)
        self._net.model.load_state_dict(weights)
        if radius > 0:
            self._net.mp = nn.MaxPool2d(radius * 2 + 1, 1, radius)
        self._net = self._net.eval().to(self.device)

    def _forward(self, img0, img1):
        out0 = self._net(img0.unsqueeze(0))[0]
        out1 = self._net(img1.unsqueeze(0))[0]
        kpts0, desc0 = out0["keypoints"], out0["descriptors"]
        kpts1, desc1 = out1["keypoints"], out1["descriptors"]
        idxs0, idxs1 = self._net.match(desc0, desc1)
        return kpts0[idxs0], kpts1[idxs1], kpts0, kpts1, desc0, desc1, None
