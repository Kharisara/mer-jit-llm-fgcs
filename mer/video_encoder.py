import os
from glob import glob

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


class VideoEncoder(nn.Module):
    """
    Simple video encoder:
    - Reads frames from a directory
    - Runs each frame through ResNet18
    - Averages frame features -> 512-dim embedding
    """

    def __init__(self, device: str = "cpu", max_frames: int = 16):
        super().__init__()
        self.device = torch.device(device)
        self.max_frames = max_frames

        # Load pretrained ResNet18 backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace final FC with identity to get 512-d features
        self.backbone.fc = nn.Identity()

        self.backbone.to(self.device)
        self.backbone.eval()

        # Standard ImageNet transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def forward(self, frames):
        """
        frames: list of PIL.Image objects
        returns: torch.Tensor of shape (512,)
        """
        if len(frames) == 0:
            # no frames -> zero vector
            return torch.zeros(512, device=self.device)

        # Apply transform to each frame
        batch = []
        for img in frames[: self.max_frames]:
            img_t = self.transform(img)
            batch.append(img_t)

        batch = torch.stack(batch, dim=0).to(self.device)  # (T, 3, 224, 224)

        # Run through backbone
        feats = self.backbone(batch)  # (T, 512)

        # Average over time
        vid_embedding = feats.mean(dim=0)  # (512,)
        return vid_embedding

    @torch.no_grad()
    def encode_frames_in_dir(self, frames_dir: str) -> torch.Tensor:
        """
        frames_dir: path to directory containing extracted frames (jpg/png)
        returns: torch.Tensor of shape (512,) on CPU
        """
        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")

        # Grab all image files
        frame_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            frame_paths.extend(glob(os.path.join(frames_dir, ext)))

        if len(frame_paths) == 0:
            raise FileNotFoundError(f"No image frames found in: {frames_dir}")

        # Sort paths for consistent ordering
        frame_paths = sorted(frame_paths)

        # Load images as PIL
        frames = []
        for p in frame_paths[: self.max_frames]:
            try:
                img = Image.open(p).convert("RGB")
                frames.append(img)
            except Exception:
                # skip unreadable frame
                continue

        if len(frames) == 0:
            # If everything failed, return zeros
            return torch.zeros(512)

        emb = self.forward(frames)       # (512,) on self.device
        return emb.detach().cpu()        # move to CPU for saving
