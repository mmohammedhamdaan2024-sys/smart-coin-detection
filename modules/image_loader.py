"""
Image loader and validator module for Smart Coin Detection System.
Handles file ingestion, format verification, EXIF correction, and dimension normalization.
"""

from typing import Union, Optional, Tuple, Dict, Any
import io
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps

from config import DEFAULT_MAX_IMAGE_DIM, SUPPORTED_EXTENSIONS


class ImageLoader:
    """Safely loads, validates, and prepares images for digital image processing."""

    def __init__(self, max_dimension: int = DEFAULT_MAX_IMAGE_DIM):
        self.max_dimension = max_dimension

    def load_image(
        self,
        source: Union[str, Path, bytes, io.BytesIO, Any],
        file_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Loads an image from a path, raw bytes, or file-like object.
        Returns a structured dictionary containing image data, metadata, and validation status.
        """
        result = {
            "is_valid": False,
            "error_message": "",
            "bgr": None,
            "rgb": None,
            "width": 0,
            "height": 0,
            "channels": 0,
            "scale_factor": 1.0,
            "original_width": 0,
            "original_height": 0,
            "file_name": file_name or "uploaded_image",
            "file_size_kb": 0.0
        }

        try:
            # 1. Handle file size and input stream
            pil_image = None
            raw_bytes = None

            if isinstance(source, (str, Path)):
                path_obj = Path(source)
                if not path_obj.exists():
                    result["error_message"] = f"File not found: {source}"
                    return result
                result["file_name"] = path_obj.name
                file_ext = path_obj.suffix.lower()
                if file_ext not in SUPPORTED_EXTENSIONS:
                    result["error_message"] = f"Unsupported file extension '{file_ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
                    return result
                result["file_size_kb"] = round(os.path.getsize(path_obj) / 1024, 2)
                pil_image = Image.open(path_obj)

            elif hasattr(source, "read"):  # Streamlit UploadedFile or BytesIO
                source.seek(0)
                raw_bytes = source.read()
                result["file_size_kb"] = round(len(raw_bytes) / 1024, 2)
                if hasattr(source, "name"):
                    result["file_name"] = source.name
                
                # Verify non-empty
                if len(raw_bytes) == 0:
                    result["error_message"] = "Uploaded file is completely empty (0 bytes)."
                    return result

                pil_image = Image.open(io.BytesIO(raw_bytes))

            elif isinstance(source, bytes):
                result["file_size_kb"] = round(len(source) / 1024, 2)
                if len(source) == 0:
                    result["error_message"] = "Byte stream is empty (0 bytes)."
                    return result
                pil_image = Image.open(io.BytesIO(source))

            elif isinstance(source, np.ndarray):
                # Already a numpy array
                img_rgb = source
                if len(img_rgb.shape) == 2:
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
                elif img_rgb.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
                
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                h, w = img_bgr.shape[:2]
                result["bgr"] = img_bgr
                result["rgb"] = img_rgb
                result["width"] = w
                result["height"] = h
                result["original_width"] = w
                result["original_height"] = h
                result["channels"] = img_bgr.shape[2]
                result["is_valid"] = True
                return result

            else:
                result["error_message"] = f"Unsupported image source type: {type(source)}"
                return result

            if pil_image is None:
                result["error_message"] = "Failed to parse image with PIL."
                return result

            # 2. Correct EXIF orientation if present
            try:
                pil_image = ImageOps.exif_transpose(pil_image)
            except Exception:
                pass  # Ignore EXIF errors if metadata is missing or corrupted

            # 3. Convert to RGB numpy array
            pil_image = pil_image.convert("RGB")
            img_rgb = np.array(pil_image)
            
            orig_h, orig_w = img_rgb.shape[:2]
            result["original_width"] = orig_w
            result["original_height"] = orig_h

            # Basic dimensions validation
            if orig_h < 20 or orig_w < 20:
                result["error_message"] = f"Image resolution ({orig_w}x{orig_h}) is too small to perform coin detection."
                return result

            # 4. Dimension scaling for performance and memory safety
            scale = 1.0
            max_side = max(orig_h, orig_w)
            if max_side > self.max_dimension:
                scale = self.max_dimension / float(max_side)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]

            result["bgr"] = img_bgr
            result["rgb"] = img_rgb
            result["width"] = w
            result["height"] = h
            result["channels"] = 3
            result["scale_factor"] = scale
            result["is_valid"] = True

        except Exception as ex:
            result["error_message"] = f"Error decoding image file: {str(ex)}"
            result["is_valid"] = False

        return result
