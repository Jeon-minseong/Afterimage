"""RMBG-2.0 (BiRefNet-based) runner — drop-in replacement for RMBGRunner.

Same interface as the v1.4 RMBGRunner so existing rmbg_mask() works unchanged.
Requires HuggingFace auth for gated `briaai/RMBG-2.0`:
  huggingface-cli login
  # or  export HF_TOKEN=hf_xxxx
"""
from __future__ import annotations
import os
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# LOCAL vendored RMBG-2.0 first (<folder>/weights/rmbg2.0/, includes birefnet.py
# for trust_remote_code), else the gated HF id (needs HF token).
_LOCAL_RMBG2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'weights', 'rmbg2.0')
DEFAULT_RMBG2_ID = (os.path.abspath(_LOCAL_RMBG2)
                    if os.path.isdir(_LOCAL_RMBG2) else 'briaai/RMBG-2.0')


class RMBG2Runner:
    """Drop-in replacement for RMBGRunner (v1.4) using BiRefNet RMBG-2.0.

    Interface:
        runner = RMBG2Runner(device='cuda:0')
        soft_mask = runner.predict(rgb_uint8_HxWx3)   # → (H, W) float32 in [0,1]
    """

    def __init__(
        self,
        device: str = 'cuda:0',
        model_id: str = None,
        input_size: Tuple[int, int] = (1024, 1024),
        use_high_precision: bool = True,
    ) -> None:
        from transformers import AutoModelForImageSegmentation

        self.device = device
        self.input_size = input_size
        model_id = model_id or DEFAULT_RMBG2_ID

        # Precision: BiRefNet benefits from float32 high matmul on Ampere+
        if use_high_precision:
            torch.set_float32_matmul_precision('high')

        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

        # ImageNet normalization (BiRefNet was trained with these)
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Predict soft foreground mask. Returns (H, W) float32 in [0,1] at
        the input image's resolution.
        """
        H, W, _ = rgb_frame.shape
        pil = Image.fromarray(rgb_frame)
        x = self.transform(pil).unsqueeze(0).to(self.device)

        preds = self.model(x)
        # BiRefNet returns a list; the LAST element is the final high-res sigmoid
        # logits. Some versions wrap in tuple. Handle both.
        if isinstance(preds, (list, tuple)):
            logits = preds[-1]
        else:
            logits = preds
        sigmoid_map = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if sigmoid_map.shape != (H, W):
            sigmoid_map = cv2.resize(sigmoid_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Min-max normalize so threshold semantics match RMBG-1.4 runner
        mn, mx = float(sigmoid_map.min()), float(sigmoid_map.max())
        if mx - mn > 1e-6:
            sigmoid_map = (sigmoid_map - mn) / (mx - mn)
        return sigmoid_map
