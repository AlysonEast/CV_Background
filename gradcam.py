# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "beartype",
#     "einops",
#     "grad-cam",
#     "jaxtyping",
#     "numpy",
#     "opencv-python",
#     "open-clip-torch",
#     "timm",
#     "torch",
#     "tyro",
# ]
# ///
import dataclasses
import io
import json
import logging
import os
import pathlib

import beartype
import cv2
import einops
import numpy as np
import open_clip
import torch
import tyro
from jaxtyping import Float, jaxtyped
from pytorch_grad_cam import (
    EigenCAM,
    EigenGradCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torch import Tensor

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("gradcam.py")

CWD = pathlib.Path(__file__).parent

methods = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
}


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    device: str = "cpu"
    """Torch device to use."""

    image_path: str = "./data/bluejay.png"
    """Input image path"""

    aug_smooth: bool = False
    """Apply test time augmentation to smooth the CAM"""

    eigen_smooth: bool = False
    """Reduce noise by taking the first principle component of cam_weights*activations"""

    method: str = "gradcam"
    """Can be gradcam/gradcam++/scorecam/xgradcam"""

    txt_emb: str = (
        "https://huggingface.co/spaces/imageomics/bioclip-demo/resolve/main/txt_emb.npy"
    )
    """Path or URL to the BioCLIP text embeddings."""

    txt_names: str = "https://huggingface.co/spaces/imageomics/bioclip-demo/resolve/main/txt_emb_species.json"
    """Path or URL to the embedded names."""


@beartype.beartype
def get_cache_dir() -> str:
    """
    Get cache directory from environment variables, defaulting to the current working directory (.)

    Returns:
        A path to a cache directory (might not exist yet).
    """
    cache_dir = ""
    for var in ("HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


@beartype.beartype
def load_model(fpath: str | pathlib.Path, *, device: str = "cpu") -> torch.nn.Module:
    """
    Loads a linear layer from disk.
    """
    with open(fpath, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())

    model = torch.nn.Linear(**kwargs)
    state_dict = torch.load(buffer, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


@jaxtyped(typechecker=beartype.beartype)
def reshape_transform(
    x_BND: Float[Tensor, "batch n_patches dim"],
) -> Float[Tensor, "batch dim width height"]:
    _, n_patches, _ = x_BND.shape

    # The -1 is for the CLS token
    n_patches_side = int((n_patches - 1) ** 0.5)

    # Bring the channels to the first dimension, like in CNNs.
    x_BWHD = einops.rearrange(
        x_BND[:, 1:, :],
        "batch (width height) dim -> batch dim width height",
        width=n_patches_side,
        height=n_patches_side,
    )

    return x_BWHD


@beartype.beartype
def maybe_download(url_or_fpath: str) -> str:
    """Checks if url_or_fpath is a file or a url, then uses torch.hub to download to cache_dir() if it's a url. Returns the path to the file on disk."""
    import os
    import pathlib
    import urllib.parse

    # Check if it's a file path that exists
    if os.path.isfile(url_or_fpath):
        return url_or_fpath

    # Check if it's a URL
    parsed = urllib.parse.urlparse(url_or_fpath)
    if parsed.scheme in ("http", "https"):
        # It's a URL, download it using torch.hub utilities
        # torch.hub.download_url_to_file handles:
        # - Creating the cache directory if it doesn't exist
        # - Checking if the file is already downloaded (avoiding redundant downloads)
        # - Handling HTTP errors and retries
        # - Progress reporting for large files
        cache_dir = pathlib.Path(get_cache_dir())
        os.makedirs(cache_dir, exist_ok=True)

        filename = os.path.basename(parsed.path)
        if not filename:
            filename = "downloaded_file"

        output_path = cache_dir / filename

        # Use torch.hub to download the file
        torch.hub.download_url_to_file(url_or_fpath, str(output_path), progress=True)

        logger.info(f"Downloaded {url_or_fpath} to {output_path}")
        return str(output_path)

    # If it's not a URL and not an existing file, assume it's a path
    return url_or_fpath


@jaxtyped(typechecker=beartype.beartype)
class Classifier(torch.nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        clip = open_clip.create_model(
            "hf-hub:imageomics/bioclip", cache_dir=get_cache_dir()
        )
        model = clip.visual
        model.output_tokens = True  # type: ignore
        self.vit = model.eval()
        self.logit_scale = clip.logit_scale

        txt_emb_fpath = maybe_download(args.txt_emb)
        self.linear = torch.from_numpy(np.load(txt_emb_fpath, mmap_mode="r"))
        self.linear = self.linear.to(args.device)

        logger.info("Loaded 'linear classifier'.")

    @property
    def blocks(self):
        return self.vit.transformer.resblocks

    def forward(
        self, x_BCWH: Float[Tensor, "batch channels width height"]
    ) -> Float[Tensor, "batch n_classes"]:
        cls_BD, _ = self.vit(x_BCWH)
        return self.logit_scale.exp() * cls_BD @ self.linear


def main():
    """
    Example usage of using cam-methods on a VIT network.
    """

    args = tyro.cli(Args)

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = Classifier(args).to(torch.device(args.device)).eval()

    target_layers = [model.blocks[-2]]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    txt_names_fpath = maybe_download(args.txt_names)
    with open(txt_names_fpath) as fd:
        txt_names = json.load(fd)

    cam = methods[args.method](
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ).to(args.device)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
        eigen_smooth=args.eigen_smooth,
        aug_smooth=args.aug_smooth,
    )

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f"{args.method}_cam.png", cam_image)


if __name__ == "__main__":
    main()
