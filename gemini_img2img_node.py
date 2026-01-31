"""
ComfyUI BananaForge Image-to-Image Node
Edit/transform images using Gemini-compatible APIs.
"""

import io
import re
import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io as comfy_io


def load_env_file() -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    node_dir = Path(__file__).parent
    env_path = node_dir / ".env"
    
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip()
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        env_vars[key] = value
        except Exception:
            pass
    return env_vars


class GeminiImg2ImgNode(comfy_io.ComfyNode):
    """
    Edit/transform images using Gemini-compatible APIs.
    Supports multiple reference images (up to 6).
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="BananaForgeImg2Img",
            display_name="🖼️ BananaForge Image → Image",
            description="Edit/transform existing images (auto-matches input aspect ratio)",
            category="Banana Forge",
            inputs=[
                # API Configuration
                comfy_io.String.Input(
                    "api_url",
                    default="https://generativelanguage.googleapis.com/v1beta",
                    multiline=False,
                    tooltip="API base URL"
                ),
                comfy_io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="API key (leave empty to use .env file)"
                ),
                comfy_io.Combo.Input(
                    "use_env_file",
                    options=["no", "yes"],
                    default="no",
                    tooltip="Load API key from .env file"
                ),
                comfy_io.String.Input(
                    "model_name",
                    default="gemini-3-pro-image-preview",
                    multiline=False,
                    tooltip="Model name"
                ),
                
                # Prompt
                comfy_io.String.Input(
                    "prompt",
                    default="Edit this image",
                    multiline=True,
                    tooltip="Instruction for image editing"
                ),
                
                # Input Images (required: at least one)
                comfy_io.Image.Input("image_1", tooltip="Primary input image (required)"),
                
                # Optional additional images
                comfy_io.Image.Input("image_2", optional=True, tooltip="Optional 2nd reference image"),
                comfy_io.Image.Input("image_3", optional=True, tooltip="Optional 3rd reference image"),
                comfy_io.Image.Input("image_4", optional=True, tooltip="Optional 4th reference image"),
                comfy_io.Image.Input("image_5", optional=True, tooltip="Optional 5th reference image"),
                comfy_io.Image.Input("image_6", optional=True, tooltip="Optional 6th reference image"),
                
                # Aspect Ratio
                comfy_io.Combo.Input(
                    "match_input_aspect",
                    options=["yes", "no"],
                    default="yes",
                    tooltip="Match output aspect ratio to input image"
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=["1:1", "16:9", "9:16", "4:3", "3:4"],
                    default="1:1",
                    tooltip="Output aspect ratio (when not matching input)"
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=["default", "2K", "4K"],
                    default="default",
                    tooltip="Output resolution"
                ),
                
                # Rate Limit
                comfy_io.Float.Input(
                    "wait_seconds",
                    default=0.0,
                    min=0.0,
                    max=60.0,
                    step=0.5,
                    tooltip="Wait time before API call (to avoid rate limits)"
                ),
            ],
            outputs=[
                comfy_io.Image.Output(display_name="IMAGE"),
            ],
            hidden=[
                comfy_io.Hidden.prompt,
                comfy_io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(
        cls,
        api_url: str,
        api_key: str,
        use_env_file: str,
        model_name: str,
        prompt: str,
        image_1: torch.Tensor,
        match_input_aspect: str,
        aspect_ratio: str,
        image_size: str,
        wait_seconds: float,
        image_2: Optional[torch.Tensor] = None,
        image_3: Optional[torch.Tensor] = None,
        image_4: Optional[torch.Tensor] = None,
        image_5: Optional[torch.Tensor] = None,
        image_6: Optional[torch.Tensor] = None,
        hidden_prompt: Optional[dict] = None,
        hidden_unique_id: Optional[str] = None,
    ) -> comfy_io.NodeOutput:
        """Execute image-to-image transformation."""
        
        # Wait to avoid rate limits
        if wait_seconds > 0:
            print(f"🖼️ [Image→Image] Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
        
        # Load from .env if requested
        if use_env_file == "yes":
            env_vars = load_env_file()
            if not api_key or not api_key.strip():
                api_key = env_vars.get("API_KEY", env_vars.get("GEMINI_API_KEY", ""))
            if not api_url or api_url == "https://generativelanguage.googleapis.com/v1beta":
                api_url = env_vars.get("API_URL", env_vars.get("GEMINI_API_URL", api_url))
        
        # Validate inputs
        if not api_url or not api_url.strip():
            raise ValueError("❌ API URL cannot be empty.")
        if not api_key or not api_key.strip():
            raise ValueError("❌ API key cannot be empty.")
        if not model_name or not model_name.strip():
            raise ValueError("❌ Model name cannot be empty.")
        if not prompt or not prompt.strip():
            raise ValueError("❌ Prompt cannot be empty.")
        
        actual_model = model_name.strip()
        is_pro_model = "pro" in actual_model.lower()
        
        # Collect all input images
        all_images = [image_1]
        for img in [image_2, image_3, image_4, image_5, image_6]:
            if img is not None:
                all_images.append(img)
        
        # Determine output aspect ratio
        if match_input_aspect == "yes":
            # Get aspect ratio from first input image
            h, w = image_1.shape[1], image_1.shape[2]
            ratio = w / h
            # Map to closest standard ratio
            ratios = {"1:1": 1.0, "16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "3:4": 3/4}
            output_aspect = min(ratios.keys(), key=lambda k: abs(ratios[k] - ratio))
            print(f"🖼️ [Image→Image] Input image {w}x{h}, matched to {output_aspect}")
        else:
            output_aspect = aspect_ratio
        
        # Add ratio/size info to prompt
        enhanced_prompt = prompt.strip()
        enhanced_prompt += f"\n[Image settings: aspect ratio {output_aspect}"
        if image_size != "default":
            enhanced_prompt += f", resolution {image_size}"
        enhanced_prompt += "]"
        
        # Build endpoint
        api_url = api_url.strip().rstrip('/')
        endpoint = f"{api_url}/models/{actual_model}:generateContent"
        
        # Build content parts
        parts = [{"text": enhanced_prompt}]
        
        # Add all reference images
        for img_tensor in all_images:
            b64_image = cls._tensor_to_base64(img_tensor)
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": b64_image
                }
            })
        
        # Build payload
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": output_aspect}
            }
        }
        
        if image_size != "default" and is_pro_model:
            payload["generationConfig"]["imageConfig"]["imageSize"] = image_size
        
        print(f"🖼️ [Image→Image] Sending to: {endpoint}")
        print(f"🖼️ [Image→Image] Config: aspect_ratio={output_aspect}, image_size={image_size}, images={len(all_images)}")
        
        # Make request
        result = cls._call_api(endpoint, api_key.strip(), payload)
        image_tensor = cls._decode_image(result["image"])
        
        print(f"🖼️ [Image→Image] Got image: {image_tensor.shape}")
        
        return comfy_io.NodeOutput(image_tensor)
    
    @classmethod
    def _tensor_to_base64(cls, tensor: torch.Tensor) -> str:
        """Convert tensor to base64 PNG."""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        np_img = (tensor.numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img, mode='RGB')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @classmethod
    def _call_api(cls, endpoint: str, api_key: str, payload: dict) -> dict:
        """Call the Gemini API."""
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        data = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(endpoint, data=data, headers=headers, method='POST')
        
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                result = {"image": None}
                
                if 'candidates' not in response_data or len(response_data['candidates']) == 0:
                    if 'promptFeedback' in response_data:
                        raise ValueError(f"❌ Content blocked: {response_data['promptFeedback'].get('blockReason', 'Unknown')}")
                    raise ValueError(f"❌ No candidates in response")
                
                candidate = response_data['candidates'][0]
                if candidate.get('finishReason') == 'SAFETY':
                    raise ValueError("❌ Content blocked by safety filters")
                
                if 'content' not in candidate or 'parts' not in candidate['content']:
                    raise ValueError("❌ No content in response")
                
                for part in candidate['content']['parts']:
                    if 'inlineData' in part and 'data' in part['inlineData']:
                        result["image"] = part['inlineData']['data']
                        break
                    elif 'text' in part:
                        # Check for image URL in markdown
                        text = part['text']
                        match = re.search(r'!\[.*?\]\((https?://[^\s\)]+)\)', text)
                        if match:
                            result["image"] = cls._download_image(match.group(1))
                            break
                
                if result["image"] is None:
                    raise ValueError("❌ No image in response")
                
                return result
                
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode('utf-8') if e.fp else str(e)
            raise ValueError(f"❌ API Error ({e.code}): {error_msg[:300]}")
        except urllib.error.URLError as e:
            raise ValueError(f"❌ Connection error: {str(e.reason)}")
    
    @classmethod
    def _download_image(cls, url: str) -> str:
        """Download image from URL and return as base64."""
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            return base64.b64encode(response.read()).decode('utf-8')
    
    @classmethod
    def _decode_image(cls, b64_data: str) -> torch.Tensor:
        """Decode base64 to tensor."""
        pil_image = Image.open(io.BytesIO(base64.b64decode(b64_data)))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
