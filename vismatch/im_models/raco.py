from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("RaCo"))
add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))

from raco import RaCo
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd


class RaCoLightGlue(BaseMatcher):
    """RaCo keypoints described with ALIKED and matched with LightGlue trained on them, as in RaCo's matching demo."""

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        # ranker and covariance heads are not used for matching: disabling them leaves keypoints unchanged
        self.extractor = RaCo(max_num_keypoints=max_num_keypoints, ranker=False, covariance_estimator=False)
        self.extractor = self.extractor.eval().to(self.device)
        self.descriptor = ALIKED().eval().to(self.device)
        self.matcher = LightGlue(features="raco-aliked", depth_confidence=-1, width_confidence=-1).to(self.device)

    def _extract(self, img):
        feats = self.extractor.extract(img)
        feats["descriptors"] = self.descriptor.describe(feats["keypoints"], img)
        return feats

    def _forward(self, img0, img1):
        feats0, feats1 = self._extract(img0), self._extract(img1)
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = rbd(feats0), rbd(feats1), rbd(matches01)

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return mkpts0, mkpts1, kpts0, kpts1, feats0["descriptors"], feats1["descriptors"], matches01["scores"]
