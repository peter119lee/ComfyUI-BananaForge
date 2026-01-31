"""
ComfyUI BananaForge Text-to-Image Node
Generate images from text prompts using Gemini-compatible APIs.
"""

import io
import re
import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

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


class GeminiText2ImgNode(comfy_io.ComfyNode):
    """
    Generate images from text prompts using Gemini-compatible APIs.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="BananaForgeText2Img",
            display_name="🍌 BananaForge Text → Image",
            description="Generate NEW images from text prompts",
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
                    default="A beautiful landscape",
                    multiline=True,
                    tooltip="Text prompt for image generation"
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
        aspect_ratio: str,
        image_size: str,
        wait_seconds: float,
        hidden_prompt: Optional[dict] = None,
        hidden_unique_id: Optional[str] = None,
    ) -> comfy_io.NodeOutput:
        """Execute text-to-image generation."""
        
        # Wait to avoid rate limits
        if wait_seconds > 0:
            print(f"🍌 [Text→Image] Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
        
        # Load from .env if requested (always override when enabled)
        if use_env_file == "yes":
            env_vars = load_env_file()
            # Always use .env values if they exist
            env_api_key = env_vars.get("API_KEY", env_vars.get("GEMINI_API_KEY", ""))
            env_api_url = env_vars.get("API_URL", env_vars.get("GEMINI_API_URL", ""))
            if env_api_key:
                api_key = env_api_key
            if env_api_url:
                api_url = env_api_url
        
        # Validate inputs
        if not api_url or not api_url.strip():
            raise ValueError("❌ API URL cannot be empty. Set it in the node or .env file.")
        if not api_key or not api_key.strip():
            raise ValueError("❌ API key cannot be empty. Set it in the node or .env file.")
        if not model_name or not model_name.strip():
            raise ValueError("❌ Model name cannot be empty.")
        if not prompt or not prompt.strip():
            raise ValueError("❌ Prompt cannot be empty.")
        
        actual_model = model_name.strip()
        is_pro_model = "pro" in actual_model.lower()
        
        # Add ratio/size info to prompt for better AI understanding
        enhanced_prompt = prompt.strip()
        enhanced_prompt += f"\n[Image settings: aspect ratio {aspect_ratio}"
        if image_size != "default":
            enhanced_prompt += f", resolution {image_size}"
        enhanced_prompt += "]"
        
        # Build endpoint
        api_url = api_url.strip().rstrip('/')
        endpoint = f"{api_url}/models/{actual_model}:generateContent"
        
        # Build payload
        payload = {
            "contents": [{"parts": [{"text": enhanced_prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio}
            }
        }
        
        if image_size != "default" and is_pro_model:
            payload["generationConfig"]["imageConfig"]["imageSize"] = image_size
        
        print(f"🍌 [Text→Image] Sending to: {endpoint}")
        print(f"🍌 [Text→Image] Config: aspect_ratio={aspect_ratio}, image_size={image_size}")
        
        # Make request
        result = cls._call_api(endpoint, api_key.strip(), payload)
        image_tensor = cls._decode_image(result["image"])
        
        print(f"🍌 [Text→Image] Got image: {image_tensor.shape}")
        
        return comfy_io.NodeOutput(image_tensor)
    
    @classmethod
    def _call_api(cls, endpoint: str, api_key: str, payload: dict, max_retries: int = 3) -> dict:
        """Call the Gemini API with retry logic."""
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        data = json.dumps(payload).encode('utf-8')
        
        last_error = None
        for attempt in range(max_retries):
            try:
                request = urllib.request.Request(endpoint, data=data, headers=headers, method='POST')
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
                last_error = f"Connection error: {str(e.reason)}"
                print(f"🍌 [Text→Image] Retry {attempt + 1}/{max_retries}: {last_error}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            except Exception as e:
                # Catch RemoteDisconnected and other transient errors
                last_error = str(e)
                print(f"🍌 [Text→Image] Retry {attempt + 1}/{max_retries}: {last_error}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise ValueError(f"❌ Failed after {max_retries} attempts: {last_error}")
    
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
