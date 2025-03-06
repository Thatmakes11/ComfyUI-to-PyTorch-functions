# ComfyUI-to-Pytorch-Function

This repo has initialized a fastapi server with multiple ComfyUI workflow tasks.

## Installation

1. Navigate to your `ComfyUI/custom_nodes` directory

2. Clone this repo
    ```bash
    git clone https://github.com/Thatmakes11/ComfyUI-to-PyTorch-functions.git
    ```

3. Move `.py` files in examples to the `ComfyUI` directory

    After cloning and moving, your `ComfyUI` directory should look like this:
    ```
    /comfy
    /comfy_extras
    /custom_nodes
    --/ComfyUI-to-Python-Extension
    /input
    /models
    /output
    /script_examples
    /web
    .gitignore
    LICENSE
    README.md
    comfyui_screenshot.png
    clothes_swap.py       # file from examples
    cuda_mollac.py
    execution.py
    extra_model_paths.yaml.example
    fastapi_server.py     # file from examples
    folder_paths.py
    latent_preview.py
    main.py
    nodes.py
    node_manager.py       # file from examples
    path_manager.py       # file from examples
    remove_background.py  # file from examples
    requirements.txt
    server.py
    ```

## Workflow to Pytorch

### CLI Usage
1. Navigate to the `ComfyUI-to-Python-Extension` folder and install requirements
    ```bash
    pip install -r requirements.txt
    ```

2. Load up your favorite workflows in ComfyUI, then click the newly enabled `Save (API Format)` button under Queue Prompt

3. Run the script with optional arguments:
   ```bash
   python comfyui_to_pytorch_function.py --input_file "workflow_api.json" --output_file my_workflow.py
   ```

4. After running `comfyui_to_pytorch_function.py`, a new `.py` file will be created in the current working directory. If you made no changes, look for `workflow_api.py`.

5. Now you have a function that execute PyTorch process.

## FastAPI server

### Usage

#### Running the API
To start the FastAPI server, run:
```sh
uvicorn fastapi_server:app --host 0.0.0.0 --port 8080
```
or
```python
python fastapi_server.py
```

#### API Endpoints
**Endpoint:** `/predict/image_generation`

**Method:** `POST`

**Request Body:**
```json
{
  "task_type": "string",
  "task_id": "string",
  "params": "dict"
}
```

Current supported tasks are **ClothesSwap** and **Rmbg**, if you wish to add your own tasks ([Workflow to Pytorch](#workflow-to-pytorch)), check for line 15 and line 52-58 in `fastapi_server.py`.

Also, your customed torch functions are required to return a dict:
```python
{
  "status": "success or failed", 
  "output_image": "Tensor or None", 
  "message": "error or completed"
}
```

**Response:**
```json
{
  "status": "success or failed", 
  "output_path": "/path/to/generated/image_{timestamp}.png or None",
  "message": "error or completed"
}
```
The result image will be saved in the "output_path".
