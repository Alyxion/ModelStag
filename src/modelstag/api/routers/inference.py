"""Inference endpoints."""

import base64
import io
import time
from typing import Optional, List

import numpy as np
from PIL import Image
from fastapi import APIRouter, Request, HTTPException, File, UploadFile
from fastapi.responses import Response

from modelstag.api.schemas.responses import InferenceResultResponse
from modelstag.core.exceptions import ModelNotFoundError, ModelNotRunningError
from modelstag.outputs.converters import mask_to_png_bytes
from modelstag.core.types import ModelStatus

router = APIRouter(prefix="/inference", tags=["inference"])


def _load_image(file: UploadFile) -> np.ndarray:
    """Load uploaded image as numpy array."""
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


@router.post("/{model_name}", response_model=InferenceResultResponse)
def run_inference(
    request: Request,
    model_name: str,
    image: UploadFile = File(...),
    output_formats: Optional[str] = "mask,polygon,bbox",
):
    """Run inference on an image."""
    manager = request.app.state.manager

    try:
        # Parse output formats
        formats = [f.strip() for f in output_formats.split(",")]

        # Load image
        img_array = _load_image(image)

        # Run inference
        start_time = time.perf_counter()
        result = manager.run_inference(model_name, img_array, formats)
        total_time = (time.perf_counter() - start_time) * 1000

        # Build response
        response = InferenceResultResponse(
            model=model_name,
            processing_time_ms=result.processing_time_ms or total_time,
        )

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        if result.mask:
            response.mask = {
                "width": result.mask.width,
                "height": result.mask.height,
                "data": result.mask.to_base64(),
                "is_segmentation": result.mask.is_segmentation,
            }

        if result.polygons:
            response.polygons = result.polygons.to_geojson()

        if result.bbox:
            response.bbox = {
                "x": result.bbox.x,
                "y": result.bbox.y,
                "width": result.bbox.width,
                "height": result.bbox.height,
            }

        return response

    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    except ModelNotRunningError:
        raise HTTPException(status_code=503, detail=f"Model not running: {model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/mask")
def get_mask(
    request: Request,
    model_name: str,
    image: UploadFile = File(...),
):
    """Run inference and return mask as PNG."""
    manager = request.app.state.manager

    try:
        # Load image
        img_array = _load_image(image)

        # Run inference
        result = manager.run_inference(model_name, img_array, ["mask"])

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        if not result.mask:
            raise HTTPException(status_code=500, detail="No mask generated")

        # Return PNG
        png_bytes = mask_to_png_bytes(result.mask.mask)
        return Response(content=png_bytes, media_type="image/png")

    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    except ModelNotRunningError:
        raise HTTPException(status_code=503, detail=f"Model not running: {model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/all-masks")
def get_sam_all_masks(
    request: Request,
    model_name: str,
    image: UploadFile = File(...),
    max_masks: int = 100,
    max_total_mb: float = 20.0,
):
    """Run SAM automatic mask generation and return all detected masks.

    Args:
        model_name: Name of the SAM model (sam, sam_large, sam_huge).
        image: Input image file.
        max_masks: Maximum number of masks to return (default: 100).
        max_total_mb: Maximum total memory for masks in MB (default: 20.0).

    Returns:
        JSON with masks as base64-encoded PNGs and metadata.
    """
    manager = request.app.state.manager

    try:
        # Check if model is available and running
        if model_name not in manager._workers:
            raise HTTPException(status_code=404, detail=f"Model not configured: {model_name}")

        state = manager._workers[model_name]
        if state.status != ModelStatus.RUNNING:
            # Try to start it
            if state.status == ModelStatus.STOPPED:
                manager.start_worker(model_name)
                # Wait for it to load
                import time as time_module
                timeout = 120
                waited = 0
                while state.status == ModelStatus.STARTING and waited < timeout:
                    time_module.sleep(0.5)
                    waited += 0.5

            if state.status != ModelStatus.RUNNING:
                raise HTTPException(status_code=503, detail=f"Model not running: {model_name}")

        # Load image
        img_array = _load_image(image)

        # Get the SAM worker and call predict_all_masks
        worker = state.direct_worker
        if not hasattr(worker, 'predict_all_masks'):
            raise HTTPException(status_code=400, detail="Model does not support all-masks")

        max_total_bytes = int(max_total_mb * 1024 * 1024)

        start_time = time.perf_counter()
        result = worker.predict_all_masks(
            img_array,
            max_masks=max_masks,
            max_total_bytes=max_total_bytes,
        )
        processing_time = (time.perf_counter() - start_time) * 1000

        masks = result['masks']
        stats = result['stats']

        # Convert masks to base64
        results = []
        for mask_data in masks:
            # Convert mask to PNG base64
            img = Image.fromarray(mask_data['mask'])
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            mask_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

            results.append({
                "mask": mask_b64,
                "area": int(mask_data['area']),
                "bbox": {
                    "x": float(mask_data['bbox'][0]),
                    "y": float(mask_data['bbox'][1]),
                    "width": float(mask_data['bbox'][2]),
                    "height": float(mask_data['bbox'][3]),
                },
                "score": float(mask_data['score']),
            })

        return {
            "model": model_name,
            "processing_time_ms": processing_time,
            "masks_count": len(results),
            "stats": stats,
            "masks": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/detect")
def run_detection(
    request: Request,
    model_name: str,
    image: UploadFile = File(...),
    classes: Optional[str] = None,
    confidence: float = 0.25,
):
    """Run object detection and return bounding boxes.

    Args:
        model_name: Name of the detection model.
        image: Input image file.
        classes: Comma-separated list of classes to detect (uses defaults if empty).
        confidence: Minimum confidence threshold (default: 0.25).
    """
    manager = request.app.state.manager

    try:
        if model_name not in manager._workers:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        state = manager._workers[model_name]
        if state.status != ModelStatus.RUNNING:
            if state.status == ModelStatus.STOPPED:
                manager.start_worker(model_name)
                import time as time_module
                timeout = 120
                waited = 0
                while state.status == ModelStatus.STARTING and waited < timeout:
                    time_module.sleep(0.5)
                    waited += 0.5

            if state.status != ModelStatus.RUNNING:
                raise HTTPException(status_code=503, detail=f"Model not running: {model_name}")

        img_array = _load_image(image)
        worker = state.direct_worker

        if not hasattr(worker, 'predict_boxes'):
            raise HTTPException(status_code=400, detail="Model does not support detection")

        class_list = [c.strip() for c in classes.split(",")] if classes else None

        start_time = time.perf_counter()
        result = worker.predict_boxes(img_array, classes=class_list, confidence=confidence)
        processing_time = (time.perf_counter() - start_time) * 1000

        return {
            "model": model_name,
            "processing_time_ms": processing_time,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/caption")
def run_caption(
    request: Request,
    model_name: str,
    image: UploadFile = File(...),
    detail_level: str = "detailed",
):
    """Generate image caption.

    Args:
        model_name: Name of the caption model.
        image: Input image file.
        detail_level: One of "brief", "detailed", "verbose" (default: "detailed").
    """
    manager = request.app.state.manager

    try:
        if model_name not in manager._workers:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        state = manager._workers[model_name]
        if state.status != ModelStatus.RUNNING:
            if state.status == ModelStatus.STOPPED:
                manager.start_worker(model_name)
                import time as time_module
                timeout = 120
                waited = 0
                while state.status == ModelStatus.STARTING and waited < timeout:
                    time_module.sleep(0.5)
                    waited += 0.5

            if state.status != ModelStatus.RUNNING:
                raise HTTPException(status_code=503, detail=f"Model not running: {model_name}")

        img_array = _load_image(image)
        worker = state.direct_worker

        if not hasattr(worker, 'caption'):
            raise HTTPException(status_code=400, detail="Model does not support captioning")

        start_time = time.perf_counter()
        result = worker.caption(img_array, detail_level=detail_level)
        processing_time = (time.perf_counter() - start_time) * 1000

        return {
            "model": model_name,
            "processing_time_ms": processing_time,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/ocr")
def run_ocr(
    request: Request,
    model_name: str,
    image: UploadFile = File(...),
):
    """Extract text from image using OCR.

    Args:
        model_name: Name of the caption model (Florence-2 supports OCR).
        image: Input image file.
    """
    manager = request.app.state.manager

    try:
        if model_name not in manager._workers:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        state = manager._workers[model_name]
        if state.status != ModelStatus.RUNNING:
            if state.status == ModelStatus.STOPPED:
                manager.start_worker(model_name)
                import time as time_module
                timeout = 120
                waited = 0
                while state.status == ModelStatus.STARTING and waited < timeout:
                    time_module.sleep(0.5)
                    waited += 0.5

            if state.status != ModelStatus.RUNNING:
                raise HTTPException(status_code=503, detail=f"Model not running: {model_name}")

        img_array = _load_image(image)
        worker = state.direct_worker

        if not hasattr(worker, 'ocr'):
            raise HTTPException(status_code=400, detail="Model does not support OCR")

        start_time = time.perf_counter()
        result = worker.ocr(img_array)
        processing_time = (time.perf_counter() - start_time) * 1000

        return {
            "model": model_name,
            "processing_time_ms": processing_time,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/pose")
def run_pose_estimation(
    request: Request,
    model_name: str,
    image: UploadFile = File(...),
):
    """Run pose estimation and return keypoints.

    Args:
        model_name: Name of the pose model (pose_mediapipe, pose_rtmo_*, pose_rtmw_*).
        image: Input image file.

    Returns:
        JSON with poses containing keypoints (x, y, confidence, name).
    """
    manager = request.app.state.manager

    try:
        if model_name not in manager._workers:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        state = manager._workers[model_name]
        if state.status != ModelStatus.RUNNING:
            if state.status == ModelStatus.STOPPED:
                manager.start_worker(model_name)
                import time as time_module
                timeout = 120
                waited = 0
                while state.status == ModelStatus.STARTING and waited < timeout:
                    time_module.sleep(0.5)
                    waited += 0.5

            if state.status != ModelStatus.RUNNING:
                raise HTTPException(status_code=503, detail=f"Model not running: {model_name}")

        img_array = _load_image(image)
        worker = state.direct_worker

        if not hasattr(worker, 'predict_keypoints'):
            raise HTTPException(status_code=400, detail="Model does not support pose estimation")

        start_time = time.perf_counter()
        result = worker.predict_keypoints(img_array)
        processing_time = (time.perf_counter() - start_time) * 1000

        return {
            "model": model_name,
            "processing_time_ms": processing_time,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/hands")
def run_hand_tracking(
    request: Request,
    model_name: str,
    image: UploadFile = File(...),
):
    """Run hand tracking and return landmarks.

    Args:
        model_name: Name of the hand model (hand_mediapipe, hand_hamer).
        image: Input image file.

    Returns:
        JSON with hands containing landmarks (x, y, z, confidence, name) and handedness.
    """
    manager = request.app.state.manager

    try:
        if model_name not in manager._workers:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        state = manager._workers[model_name]
        if state.status != ModelStatus.RUNNING:
            if state.status == ModelStatus.STOPPED:
                manager.start_worker(model_name)
                import time as time_module
                timeout = 120
                waited = 0
                while state.status == ModelStatus.STARTING and waited < timeout:
                    time_module.sleep(0.5)
                    waited += 0.5

            if state.status != ModelStatus.RUNNING:
                raise HTTPException(status_code=503, detail=f"Model not running: {model_name}")

        img_array = _load_image(image)
        worker = state.direct_worker

        if not hasattr(worker, 'predict_landmarks'):
            raise HTTPException(status_code=400, detail="Model does not support hand tracking")

        start_time = time.perf_counter()
        result = worker.predict_landmarks(img_array)
        processing_time = (time.perf_counter() - start_time) * 1000

        return {
            "model": model_name,
            "processing_time_ms": processing_time,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
