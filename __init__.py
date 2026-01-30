"""
ComfyUI Nano Banana Extension
Three nodes for Gemini image generation:
- 🍌 Nano Banana Text → Image: Generate images from text
- 🖼️ Nano Banana Image → Image: Edit/transform existing images  
- 🍌 Nano Banana Batch: Multiple prompts → multiple images
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .gemini_text2img_node import GeminiText2ImgNode
from .gemini_img2img_node import GeminiImg2ImgNode
from .gemini_batch_node import GeminiBatchNode


class CustomImageAPIExtension(ComfyExtension):
    """Extension class for Nano Banana nodes."""
    
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GeminiText2ImgNode,
            GeminiImg2ImgNode,
            GeminiBatchNode,
        ]


async def comfy_entrypoint() -> CustomImageAPIExtension:
    """ComfyUI calls this to load the extension and its nodes."""
    return CustomImageAPIExtension()


# Legacy support for older ComfyUI versions
NODE_CLASS_MAPPINGS = {
    "NanoBananaText2Img": GeminiText2ImgNode,
    "NanoBananaImg2Img": GeminiImg2ImgNode,
    "NanoBananaBatch": GeminiBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaText2Img": "🍌 Nano Banana Text → Image",
    "NanoBananaImg2Img": "🖼️ Nano Banana Image → Image",
    "NanoBananaBatch": "🍌 Nano Banana Batch (Multi-Prompt)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'comfy_entrypoint']
