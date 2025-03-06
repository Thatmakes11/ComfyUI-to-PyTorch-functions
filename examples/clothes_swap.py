import random
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


from nodes import (
    VAELoader,
    CLIPTextEncode,
    CLIPVisionLoader,
    ConditioningZeroOut,
    UNETLoader,
    InpaintModelConditioning,
    KSampler,
    VAEDecode,
    DualCLIPLoader,
    StyleModelLoader,
    NODE_CLASS_MAPPINGS,
    EmptyImage,
)

from fastapi_server import LoadImage, get_value_at_index

@torch.no_grad()
def clothes_swap(source_image="source_image.png", target_image="target_image.png"):
    """
    Perform clothes swap using deep learning models.

    Args:
        source_image (str): Path to the source image (image of a person).
        target_image (str): Path to the target image (image with desired clothing).

    Returns:
        Dict: A dictionary containing the generated image.

    Raises:
        FileNotFoundError: If the source or target image is not found.
        KeyError: If a required model is missing in NODE_CLASS_MAPPINGS.
        RuntimeError: If inference fails due to model issues (e.g., CUDA out of memory).
        Exception: For any other unexpected errors.
    """
    try:
        # Check if the image files exist
        import os
        if not os.path.exists(source_image):
            raise FileNotFoundError(f"Source image not found: {source_image}")
        if not os.path.exists(target_image):
            raise FileNotFoundError(f"Target image not found: {target_image}")
        
        # Load required custom nodes and models
        import_custom_nodes()
        dualcliploader = DualCLIPLoader()
        dualcliploader_40 = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp8_e4m3fn.safetensors",
            type="flux",
            device="default",
        )

        unetloader = UNETLoader()
        unetloader_42 = unetloader.load_unet(
            unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn"
        )

        vaeloader = VAELoader()
        vaeloader_45 = vaeloader.load_vae(vae_name="ae.safetensors")

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_58 = clipvisionloader.load_clip(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )

        stylemodelloader = StyleModelLoader()
        stylemodelloader_59 = stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_142 = cliptextencode.encode(
            text="", clip=get_value_at_index(dualcliploader_40, 0)
        )

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_92 = fluxguidance.append(
            guidance=30, conditioning=get_value_at_index(cliptextencode_142, 0)
        )

        loadimage = LoadImage()
        loadimage_84 = loadimage.load_image(image=target_image)

        layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS[
            "LayerUtility: ImageScaleByAspectRatio V2"
        ]()
        layerutility_imagescalebyaspectratio_v2_87 = (
            layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="letterbox",
                method="lanczos",
                round_to_multiple="None",
                scale_to_side="longest",
                scale_to_length=1024,
                background_color="#000000",
                image=get_value_at_index(loadimage_84, 0),
            )
        )

        apersonmaskgenerator = NODE_CLASS_MAPPINGS["APersonMaskGenerator"]()
        apersonmaskgenerator_127 = apersonmaskgenerator.generate_mask(
            face_mask=False,
            background_mask=False,
            hair_mask=False,
            body_mask=False,
            clothes_mask=True,
            confidence=0.9,
            refine_mask=True,
            images=get_value_at_index(layerutility_imagescalebyaspectratio_v2_87, 0),
        )

        reduxadvanced = NODE_CLASS_MAPPINGS["ReduxAdvanced"]()
        reduxadvanced_65 = reduxadvanced.apply_stylemodel(
            downsampling_factor=1,
            downsampling_function="area",
            mode="autocrop with mask",
            weight=1,
            autocrop_margin=0.1,
            conditioning=get_value_at_index(fluxguidance_92, 0),
            style_model=get_value_at_index(stylemodelloader_59, 0),
            clip_vision=get_value_at_index(clipvisionloader_58, 0),
            image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_87, 0),
            mask=get_value_at_index(apersonmaskgenerator_127, 0),
        )

        conditioningzeroout = ConditioningZeroOut()
        conditioningzeroout_91 = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cliptextencode_142, 0)
        )

        loadimage_90 = loadimage.load_image(image=source_image)

        layerutility_imagescalebyaspectratio_v2_89 = (
            layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="letterbox",
                method="lanczos",
                round_to_multiple="None",
                scale_to_side="longest",
                scale_to_length=1024,
                background_color="#000000",
                image=get_value_at_index(loadimage_90, 0),
            )
        )

        layermask_maskinvert = NODE_CLASS_MAPPINGS["LayerMask: MaskInvert"]()
        layermask_maskinvert_131 = layermask_maskinvert.mask_invert(
            mask=get_value_at_index(apersonmaskgenerator_127, 0)
        )

        imageandmaskpreview = NODE_CLASS_MAPPINGS["ImageAndMaskPreview"]()
        imageandmaskpreview_129 = imageandmaskpreview.execute(
            mask_opacity=1,
            mask_color="0,0,0",
            pass_through=True,
            image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_87, 0),
            mask=get_value_at_index(layermask_maskinvert_131, 0),
        )

        imageconcanate = NODE_CLASS_MAPPINGS["ImageConcanate"]()
        imageconcanate_96 = imageconcanate.concatenate(
            direction="right",
            match_image_size=True,
            image1=get_value_at_index(layerutility_imagescalebyaspectratio_v2_89, 0),
            image2=get_value_at_index(imageandmaskpreview_129, 0),
        )

        apersonmaskgenerator_133 = apersonmaskgenerator.generate_mask(
            face_mask=False,
            background_mask=False,
            hair_mask=False,
            body_mask=False,
            clothes_mask=True,
            confidence=0.9,
            refine_mask=True,
            images=get_value_at_index(layerutility_imagescalebyaspectratio_v2_89, 0),
        )

        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        masktoimage_97 = masktoimage.mask_to_image(
            mask=get_value_at_index(apersonmaskgenerator_133, 0)
        )

        emptyimage = EmptyImage()
        emptyimage_95 = emptyimage.generate(
            width=get_value_at_index(layerutility_imagescalebyaspectratio_v2_87, 3),
            height=get_value_at_index(layerutility_imagescalebyaspectratio_v2_87, 4),
            batch_size=1,
            color=0,
        )

        imageconcanate_98 = imageconcanate.concatenate(
            direction="right",
            match_image_size=True,
            image1=get_value_at_index(masktoimage_97, 0),
            image2=get_value_at_index(emptyimage_95, 0),
        )

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_108 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(imageconcanate_98, 0)
        )

        inpaintmodelconditioning = InpaintModelConditioning()
        inpaintmodelconditioning_83 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(reduxadvanced_65, 0),
            negative=get_value_at_index(conditioningzeroout_91, 0),
            vae=get_value_at_index(vaeloader_45, 0),
            pixels=get_value_at_index(imageconcanate_96, 0),
            mask=get_value_at_index(imagetomask_108, 0),
        )

        differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()

        differentialdiffusion_41 = differentialdiffusion.apply(
            model=get_value_at_index(unetloader_42, 0)
        )

        ksampler_43 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=1,
            cfg=1,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(differentialdiffusion_41, 0),
            positive=get_value_at_index(inpaintmodelconditioning_83, 0),
            negative=get_value_at_index(inpaintmodelconditioning_83, 1),
            latent_image=get_value_at_index(inpaintmodelconditioning_83, 2),
        )

        vaedecode_47 = vaedecode.decode(
            samples=get_value_at_index(ksampler_43, 0),
            vae=get_value_at_index(vaeloader_45, 0),
        )

        imagecrop_105 = imagecrop.execute(
            width=get_value_at_index(layerutility_imagescalebyaspectratio_v2_89, 3),
            height=get_value_at_index(
                layerutility_imagescalebyaspectratio_v2_89, 4
            ),
            position="top-left",
            x_offset=0,
            y_offset=0,
            image=get_value_at_index(vaedecode_47, 0),
        )

        image = get_value_at_index(imagecrop_105, 0)
        return {"source_image": source_image, "target_image": target_image, "output_image": image}

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))  # 400: Bad Request (Missing File)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")  # 500: Internal Server Error
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")  # 500: Internal Server Error
    except Exception as e:
        error_info = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Unknown error: {error_info}")  # 500: Internal Server Error
