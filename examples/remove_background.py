import torch

from nodes import NODE_CLASS_MAPPINGS
from path_manager import get_value_at_index, handle_exception
from node_manager import LoadImage, import_custom_nodes


@torch.no_grad()
def remove_background(image="image.png"):
    try:
        # Check if the image file exists
        import os
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")

        import_custom_nodes()

        with torch.inference_mode():
            loadimage = LoadImage()
            loadimage_1 = loadimage.load_image(image=image)

            birefnet = NODE_CLASS_MAPPINGS["BiRefNet"]()
            if birefnet is None:
                raise KeyError("BiRefNet model not found in NODE_CLASS_MAPPINGS")

            birefnet_2 = birefnet.process_image(
                model="BiRefNet-general",
                mask_blur=0,
                mask_offset=0,
                background="Alpha",
                invert_output=False,
                refine_foreground=False,
                image=get_value_at_index(loadimage_1, 0),
            )

            image = get_value_at_index(birefnet_2, 0)
            return {"status": "success", "output_image": image, "message": "Image background removed successfully"}

    except Exception as e:
        results = handle_exception(e)
        return {"status": "failed", "output_image": None, "message": results["message"]}
