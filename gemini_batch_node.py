"""
ComfyUI BananaForge Batch Node
Process multiple prompts at once, generating one image per prompt.
Supports both text-to-image and image-to-image modes.
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


class GeminiBatchNode(comfy_io.ComfyNode):
    """
    Process multiple prompts - one image per line.
    Supports both text-to-image (no input image) and image-to-image (with reference).
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="BananaForgeBatch",
            display_name="🍌 BananaForge Batch (Multi-Prompt)",
            description="Generate multiple images from multiple prompts (one per line)",
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
                
                # Multi-line Prompts (one per line)
                comfy_io.String.Input(
                    "prompts",
                    default="A girl sitting on a chair\nA girl eating an apple\nA girl standing by the window",
                    multiline=True,
                    tooltip="Multiple prompts - ONE PER LINE. Each line generates one image."
                ),
                
                # Optional Reference Image for I2I mode
                comfy_io.Image.Input(
                    "reference_image",
                    optional=True,
                    tooltip="Optional: Reference image for image-to-image mode (used for ALL prompts)"
                ),
                
                # Image Settings
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=["1:1", "16:9", "9:16", "4:3", "3:4"],
                    default="1:1",
                    tooltip="Output aspect ratio"
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=["default", "2K", "4K"],
                    default="default",
                    tooltip="Output resolution"
                ),
                
                # Rate Limit
                comfy_io.Float.Input(
                    "wait_between",
                    default=2.0,
                    min=0.0,
                    max=60.0,
                    step=0.5,
                    tooltip="Wait time between API calls (to avoid rate limits)"
                ),
            ],
            outputs=[
                comfy_io.Image.Output(display_name="IMAGES"),
                comfy_io.Int.Output(display_name="COUNT"),
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
        prompts: str,
        aspect_ratio: str,
        image_size: str,
        wait_between: float,
        reference_image: Optional[torch.Tensor] = None,
        hidden_prompt: Optional[dict] = None,
        hidden_unique_id: Optional[str] = None,
    ) -> comfy_io.NodeOutput:
        """Execute batch image generation."""
        
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
        if not prompts or not prompts.strip():
            raise ValueError("❌ Prompts cannot be empty.")
        
        actual_model = model_name.strip()
        is_pro_model = "pro" in actual_model.lower()
        
        # Parse prompts - one per line, skip empty lines
        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        
        if len(prompt_list) == 0:
            raise ValueError("❌ No valid prompts found. Add prompts one per line.")
        
        # Determine mode
        mode = "img2img" if reference_image is not None else "txt2img"
        print(f"🍌 [Batch] Mode: {mode.upper()}, Prompts: {len(prompt_list)}")
        
        # Prepare reference image if provided
        ref_b64 = None
        if reference_image is not None:
            ref_b64 = cls._tensor_to_base64(reference_image)
            print(f"🍌 [Batch] Reference image prepared")
        
        # Build endpoint
        api_url = api_url.strip().rstrip('/')
        endpoint = f"{api_url}/models/{actual_model}:generateContent"
        
        # Process each prompt
        generated_images = []
        
        for i, prompt in enumerate(prompt_list):
            print(f"🍌 [Batch] Processing {i+1}/{len(prompt_list)}: {prompt[:50]}...")
            
            # Wait between calls (except first)
            if i > 0 and wait_between > 0:
                print(f"🍌 [Batch] Waiting {wait_between}s...")
                time.sleep(wait_between)
            
            # Add ratio/size info to prompt
            enhanced_prompt = prompt
            enhanced_prompt += f"\n[Image settings: aspect ratio {aspect_ratio}"
            if image_size != "default":
                enhanced_prompt += f", resolution {image_size}"
            enhanced_prompt += "]"
            
            # Build content parts
            if ref_b64:
                # Image-to-image mode
                parts = [
                    {"text": enhanced_prompt},
                    {"inlineData": {"mimeType": "image/png", "data": ref_b64}}
                ]
            else:
                # Text-to-image mode
                parts = [{"text": enhanced_prompt}]
            
            # Build payload
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio}
                }
            }
            
            if image_size != "default" and is_pro_model:
                payload["generationConfig"]["imageConfig"]["imageSize"] = image_size
            
            try:
                result = cls._call_api(endpoint, api_key.strip(), payload)
                image_tensor = cls._decode_image(result["image"])
                generated_images.append(image_tensor)
                print(f"🍌 [Batch] ✓ Image {i+1} generated")
            except Exception as e:
                print(f"🍌 [Batch] ✗ Image {i+1} failed: {str(e)[:100]}")
                # Continue with other prompts
        
        if len(generated_images) == 0:
            raise ValueError("❌ All prompts failed. Check API key and prompts.")
        
        # Combine all images into a batch tensor
        combined = torch.cat(generated_images, dim=0)
        print(f"🍌 [Batch] Complete: {len(generated_images)}/{len(prompt_list)} images generated")
        
        return comfy_io.NodeOutput(combined, len(generated_images))
    
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
                        raise ValueError(f"Content blocked: {response_data['promptFeedback'].get('blockReason', 'Unknown')}")
                    raise ValueError(f"No candidates in response")
                
                candidate = response_data['candidates'][0]
                if candidate.get('finishReason') == 'SAFETY':
                    raise ValueError("Content blocked by safety filters")
                
                if 'content' not in candidate or 'parts' not in candidate['content']:
                    raise ValueError("No content in response")
                
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
                    raise ValueError("No image in response")
                
                return result
                
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode('utf-8') if e.fp else str(e)
            raise ValueError(f"API Error ({e.code}): {error_msg[:200]}")
        except urllib.error.URLError as e:
            raise ValueError(f"Connection error: {str(e.reason)}")
    
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
