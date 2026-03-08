"""Model workers."""

from modelstag.workers.base import BaseModelWorker
from modelstag.workers.registry import WorkerRegistry, get_worker_class
from modelstag.workers.rembg_worker import RembgWorker
from modelstag.workers.sam_worker import SamWorker
from modelstag.workers.depth_worker import DepthWorker
from modelstag.workers.detect_worker import DetectWorker
from modelstag.workers.caption_worker import CaptionWorker
from modelstag.workers.pose_worker import PoseWorker
from modelstag.workers.hand_worker import HandWorker

__all__ = [
    "BaseModelWorker",
    "WorkerRegistry",
    "get_worker_class",
    "RembgWorker",
    "SamWorker",
    "DepthWorker",
    "DetectWorker",
    "CaptionWorker",
    "PoseWorker",
    "HandWorker",
]
