import torch
import traceback
from fastapi import HTTPException


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()

from nodes import NODE_CLASS_MAPPINGS
from fastapi_server import LoadImage, get_value_at_index

@torch.no_grad()
def mask_generate(image="image.png"):
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
            return {"output_image": image}

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))  # 400: Bad Request (Missing File)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")  # 500: Internal Server Error
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")  # 500: Internal Server Error
    except Exception as e:
        error_info = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Unknown error: {error_info}")  # 500: Internal Server Error
