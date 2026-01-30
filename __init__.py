"""
ComfyUI-BananaForge
Gemini-Powered Image Generation for ComfyUI
- 🍌 BananaForge Text → Image: Generate images from text
- 🖼️ BananaForge Image → Image: Edit/transform existing images  
- 🍌 BananaForge Batch: Multiple prompts → multiple images
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .gemini_text2img_node import GeminiText2ImgNode
from .gemini_img2img_node import GeminiImg2ImgNode
from .gemini_batch_node import GeminiBatchNode


class BananaForgeExtension(ComfyExtension):
    """Extension class for BananaForge nodes."""
    
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GeminiText2ImgNode,
            GeminiImg2ImgNode,
            GeminiBatchNode,
        ]


async def comfy_entrypoint() -> BananaForgeExtension:
    """ComfyUI calls this to load the extension and its nodes."""
    return BananaForgeExtension()


# Legacy support for older ComfyUI versions
NODE_CLASS_MAPPINGS = {
    "BananaForgeText2Img": GeminiText2ImgNode,
    "BananaForgeImg2Img": GeminiImg2ImgNode,
    "BananaForgeBatch": GeminiBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BananaForgeText2Img": "🍌 BananaForge Text → Image",
    "BananaForgeImg2Img": "🖼️ BananaForge Image → Image",
    "BananaForgeBatch": "🍌 BananaForge Batch (Multi-Prompt)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'comfy_entrypoint']
