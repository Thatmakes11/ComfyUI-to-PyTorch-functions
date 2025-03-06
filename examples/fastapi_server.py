from fastapi import FastAPI
from pydantic import BaseModel, model_validator
from typing import Any, Dict

from path_manager import add_comfyui_directory_to_sys_path, add_extra_model_paths
from node_manager import SaveImage

add_comfyui_directory_to_sys_path()
add_extra_model_paths()


# Create FastAPI application
app = FastAPI()

SUPPORED_TASK_TYPES = ["ClothesSwap", "Rmbg"]

# Define the request body data model
class TaskRequest(BaseModel):
    task_type: str
    task_id: str
    params: Dict[str, Any]  # params as a dictionary

    @model_validator(mode='before')
    def validate_params(cls, values):
        task_type = values.get("task_type")
        params = values.get("params", {})

        if task_type == "ClothesSwap":
            if "source_image" not in params or not isinstance(params["source_image"], str):
                raise ValueError(f"Missing or invalid 'source_image' for {task_type}")
            if "target_image" not in params or not isinstance(params["target_image"], str):
                raise ValueError(f"Missing or invalid 'target_image' for {task_type}")

        elif task_type == "Rmbg":
            if "image" not in params or not isinstance(params["image"], str):
                raise ValueError(f"Missing or invalid 'image' for {task_type}")

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return values


@app.post("/predict/image_generation")
async def predict_image_generation(request: TaskRequest):

    task_type = request.task_type
    task_id = request.task_id
    params = request.params

    # execute task based on task_type
    if task_type == "ClothesSwap":
        from clothes_swap import clothes_swap
        results = clothes_swap(params["source_image"], params["target_image"])

    elif task_type == "Rmbg":
        from remove_background import remove_background
        results = remove_background(params["image"])

    if results["status"] == "failed":
        return {"status": "failed", "output_path": None, "message": results["message"]}

    # save the returned image to the output directory
    result_tensors = results["output_image"]
    saveimage = SaveImage()
    output_path = saveimage.save_images(images=result_tensors, 
                                        subfolder=task_type, 
                                        task_id=task_id
                                        )[-1]["absolute_path"]
    
    return {"status": "success", "output_path": output_path, "message": results["message"]}


# To run the FastAPI app, use the following command:
# python remove_background.py
# or
# uvicorn fastapi_server:app --host 0.0.0.0 --port 8080


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)

