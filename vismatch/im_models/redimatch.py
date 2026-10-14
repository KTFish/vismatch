import torch
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import to_pil_image

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("REDI-Match"))

from redimatch.models.model_builder import get_model, install_gp16_cholesky_solver


class REDIMatchMatcher(BaseMatcher):
    """Dense matcher with a rotation-equivariant encoder, built and configured as in REDI-Match's demo."""

    def __init__(self, device="cpu", max_num_keypoints=2048, weights="outdoor", *args, **kwargs):
        super().__init__(device, **kwargs)
        self.max_num_keypoints = max_num_keypoints

        self.model = get_model(resolution="medium", symmetric=True, upsample_preds=False, attenuate_cert=False)
        install_gp16_cholesky_solver(self.model)
        checkpoint = torch.load(hf_hub_download("YinjiGe/REDI-Match", f"{weights}.pth"), map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.eval().to(self.device)

        # C4 rotation matching, with the demo's settings
        decoder = self.model.decoder
        decoder.c4_rotation_matching = True
        decoder.c4_perm_reverse = True
        decoder.c4_symmetric_reverse = "align_to_query"
        decoder.c4_hybrid = True
        decoder.c4_hybrid_alpha = 0.35
        decoder.c4_hybrid_temp = 0.05
        decoder.c4_hybrid_margin_keep = 0.03
        decoder.c4_hybrid_disagree_margin = 0.03
        decoder.c4_hybrid_ot_pool_size = 12
        decoder.c4_hybrid_ot_epsilon = 0.07
        decoder.c4_hybrid_ot_iters = 10

    def _forward(self, img0, img1):
        (h0, w0), (h1, w1) = img0.shape[-2:], img1.shape[-2:]
        # the model resizes PIL images to its resolution and normalizes them, as in the demo
        warp, certainty = self.model.match(to_pil_image(img0.cpu()), to_pil_image(img1.cpu()), device=self.device)
        matches, mconf = self.model.sample(warp, certainty, self.max_num_keypoints)
        mkpts0, mkpts1 = self.model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return mkpts0, mkpts1, None, None, None, None, mconf
