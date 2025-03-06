import os
import sys
from typing import Sequence, Mapping, Any, Union, Dict
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


from PIL import Image, ImageOps, ImageSequence
import numpy as np
import hashlib
import requests
from io import BytesIO

import folder_paths
import node_helpers


class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = image

        # Load the image
        if image.startswith('http://') or image.startswith('https://'):
            image_path = image
            response = requests.get(image)
            img = node_helpers.pillow(Image.open, BytesIO(response.content))
        elif os.path.exists(image):
            image_path = image
            img = node_helpers.pillow(Image.open, image_path)
        else:
            raise ValueError("Invalid input: neither a valid URL nor a local file path.")

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "subfolder": ("STRING", {"default": "ComfyUI", "tooltip": "The subfolder for the file to save."}),
                "task_id": ("STRING", {"default": "task_id", "tooltip": "The task ID."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, subfolder, task_id):
        subfolder += self.prefix_append
        full_output_folder, subfolder = (folder_paths.get_save_image_path(
            subfolder, self.output_dir, images[0].shape[1], images[0].shape[0]
            )[i] for i in [0, 4])

        if not os.path.exists(os.path.join(full_output_folder, subfolder)):
            os.mkdir(os.path.join(full_output_folder, subfolder))

        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            file = f"{task_id}.png"
            img.save(os.path.join(full_output_folder, subfolder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "absolute_path": os.path.join(full_output_folder, subfolder, file),
                "subfolder": subfolder,
                "type": self.type
            })

        return results


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Create FastAPI application
app = FastAPI()

SUPPORED_TASKS = ["ClothesSwap", "Rmbg"]

# Define the request body data model
class TaskRequest(BaseModel):
    task: str  # task type
    task_id: str  # task ID
    params: Union[str, Dict[str, Any]]  # params as a JSON or dictionary


@app.post("/predict/image_generation")
async def predict_image_generation(request: TaskRequest):
    """
    Unified support for:
    1. Task type `task`
    2. Task ID `task_id`
    3. `params` parsed as a dictionary
    """

    # parse JSON string
    import json
    if isinstance(request.params, str): # parse JSON string
        params = json.loads(request.params)
    else: # already a dictionary
        params = request.params

    # execute task
    task = request.task
    task_id = request.task_id

    if task == "ClothesSwap":
        if len(params) != 2:
            raise HTTPException(status_code=400, detail="ClothesSwap task supports 2 params {'source_image', 'target_image'}")
        from clothes_swap import clothes_swap
        execute = clothes_swap

    elif task == "Rmbg":
        if len(params) != 1:
            raise HTTPException(status_code=400, detail="Rmbg task supports 1 param {'image'}")
        from remove_background import mask_generate
        execute = mask_generate

    elif task not in SUPPORED_TASKS:
        raise HTTPException(status_code=400, detail=f"Unsupported task type: {task}, please use {SUPPORED_TASKS}")
    
    # save the returned image to the output directory
    result_tensors = execute(**params)["output_image"]

    saveimage = SaveImage()
    output_path = saveimage.save_images(images=result_tensors, 
                                        subfolder=task, 
                                        task_id=task_id
                                        )[-1]["absolute_path"]

    return {"output_path": output_path}


# To run the FastAPI app, use the following command:
# python remove_background.py
# or
# uvicorn fastapi_server:app --host 0.0.0.0 --port 8080


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)

