import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch.nn.functional as F
import cv2
import gc
import tempfile

# Обновляем путь к sam2
SAM2_PATH = os.path.join(os.path.dirname(__file__), "samurai", "sam2", "sam2")
if SAM2_PATH not in sys.path:
    sys.path.insert(0, SAM2_PATH)

# Обновляем путь к utils
UTILS_PATH = os.path.join(SAM2_PATH, "utils")
if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

def cleanup_memory():
    """Очистка памяти CUDA"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class SAMURAIBoxInputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "display": "number"
                }),
                "refresh_input": ("INT", {
                    "default": 0,
                    "min": 0,
                    "display": "number"
                })
            }
        }
    
    RETURN_TYPES = ("BOX", "START_FRAME")  # Изменили INT на START_FRAME
    FUNCTION = "get_box"
    CATEGORY = "SAMURAI"  
    DESCRIPTION = """# SAMURAI Box Input Node
    
This node allows you to select a region of interest (box) in the first frame of a video sequence.

## Inputs:
- **image**: Input video frames sequence (required)
- **start_frame**: Starting frame number for processing (default: 0)
- **refresh_input**: Trigger to refresh box selection (default: 0)

## Outputs:
- **BOX**: Selected bounding box coordinates
- **START_FRAME**: Frame number to start processing from

## Usage:
1. Connect video input
2. Set starting frame number
3. Draw a box around the object you want to track
4. Box coordinates will be passed to SAMURAI Refine node"""

    def get_box(self, image, start_frame=0, refresh_input=0):
        print(f"Getting box for frame {start_frame} (refresh state: {refresh_input})")
        # Подготовка изображения
        frame = image[start_frame].cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Создаем новое окно каждый раз с уникальным именем
        window_name = f'Select Box - Frame {start_frame} (Refresh: {refresh_input}) - Press ENTER when done, ESC to cancel'
        cv2.destroyAllWindows()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        box = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        
        cleanup_memory()
        return (box, start_frame)  # Возвращаем и box, и start_frame

class SAMURAIPointsInputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "start_frame": ("START_FRAME", {
                    "default": 0,
                    "min": 0,
                    "display": "number",
                    "tooltip": "Select the starting frame for points input"
                }),
                "refresh_input": ("INT", {
                    "default": 0,
                    "min": 0,
                    "display": "number",
                    "tooltip": "Change this value to select new points"
                })
            }
        }
    
    RETURN_TYPES = ("POINTS", "LABELS", "START_FRAME")  # INT для start_frame
    FUNCTION = "get_points"
    CATEGORY = "SAMURAI"
    DESCRIPTION = """# SAMURAI Points Input Node
    
This node allows you to select points of interest in the first frame of a video sequence for object segmentation.

## Inputs:
- **image**: Input video frames sequence (required)
- **start_frame**: Starting frame number for processing (default: 0)
- **refresh_input**: Trigger to refresh points selection (default: 0)

## Outputs:
- **POINTS**: Selected point coordinates for segmentation
- **LABELS**: Point labels (positive/negative)
- **START_FRAME**: Frame number to start processing from

## Usage:
1. Connect video input
2. Set starting frame number
3. Left click to add positive points (object)
4. Right click to add negative points (background)
5. Points and their labels will be passed to SAMURAI Refine node

## Note:
- Positive points (left click) indicate areas that belong to the object
- Negative points (right click) indicate areas that belong to the background"""

    def get_points(self, image, start_frame=0, refresh_input=0):
        frame = image[start_frame].cpu().numpy()
        # ... остальной код ...
        return (points_array, labels_array, start_frame)

    def get_points(self, image, start_frame=0, refresh_input=0):
        # Подготовка изображения
        frame = image[start_frame].cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        points = []
        labels = []
        frame_copy = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal frame_copy, points, labels
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                labels.append(1)
                cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow(window_name, frame_copy)
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.append([x, y])
                labels.append(0)
                cv2.circle(frame_copy, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow(window_name, frame_copy)
        
        window_name = f'Select Points - Left: positive, Right: negative, ENTER: done, ESC: cancel (Refresh: {refresh_input})'
        cv2.destroyAllWindows()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, frame_copy)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 27:  # Esc
                points = []
                labels = []
                break
        
        cv2.destroyAllWindows()
        
        points_array = np.array(points) if points else None
        labels_array = np.array(labels) if labels else None
        
        cleanup_memory()
        return (points_array, labels_array)

class SAMURAIRefineNode:
    @classmethod
    def get_config_path(cls, config_name):
        return os.path.join(os.path.dirname(__file__), "samurai", "sam2", "sam2", "configs", "samurai", config_name)

    MODEL_CONFIGS = {
        "sam2.1_hiera_base.pt": "sam2.1_hiera_t",
        "sam2.1_hiera_base_plus.pt": "sam2.1_hiera_b+",
        "sam2.1_hiera_large.pt": "sam2.1_hiera_l",
        "sam2.1_hiera_small.pt": "sam2.1_hiera_s"
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (list(cls.MODEL_CONFIGS.keys()), {
                    "default": "sam2.1_hiera_base_plus.pt"
                }),
                "resolution": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 2048,
                    "step": 8
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                })
            },
            "optional": {
                "box": ("BOX",),
                "points": ("POINTS",),
                "labels": ("LABELS",),
                "start_frame": ("START_FRAME",)
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "segment"
    CATEGORY = "SAMURAI"
    DESCRIPTION = """# SAMURAI Refine Node

This node performs video object segmentation using the SAMURAI model.

## Inputs:
- **image**: Input video frames sequence (required)
- **model_name**: SAMURAI model to use (required)
- **resolution**: Maximum resolution for processing (default: 1024)
- **iou_threshold**: Intersection over Union threshold (default: 0.1)
- **box**: Bounding box from SAMURAI Box Input (optional)
- **points**: Point prompts for segmentation (optional)
- **labels**: Labels for point prompts (optional)
- **start_frame**: Starting frame number (optional)

## Outputs:
- **MASK**: Generated segmentation masks
- **frame_number**: Current frame number for sequence

## Models Available:
- sam2.1_hiera_base.pt
- sam2.1_hiera_base_plus.pt
- sam2.1_hiera_large.pt
- sam2.1_hiera_small.pt

## Usage:
1. Connect video and box/points inputs
2. Select desired model and parameters
3. Node will generate masks for the selected object through the video sequence"""

    def __init__(self):
        self.model = None
        self.predictor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_state()

    def reset_state(self):
        """Сброс состояния узла"""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        cleanup_memory()

    def load_model(self, model_name):
        if self.predictor is None:
            print("SAMURAI mode: True")
            from sam2.build_sam import build_sam2_video_predictor
            
            config_name = self.MODEL_CONFIGS.get(model_name)
            if not config_name:
                raise ValueError(f"Configuration not found for model {model_name}")
            
            config_file = self.get_config_path(config_name)
            models_path = os.path.join(os.path.dirname(__file__), 
                                    "samurai", "sam2", "checkpoints")
            model_path = os.path.join(models_path, model_name)
            
            self.predictor = build_sam2_video_predictor(
                config_file=config_file,
                ckpt_path=model_path
            )
            print(f"Loaded checkpoint {model_name} with config {config_file}")

    def segment(self, image, model_name, resolution=1024, iou_threshold=0.1, box=None, points=None, labels=None, start_frame=0):
        self.reset_state()
        self.load_model(model_name)

        if not isinstance(image, torch.Tensor):
            raise ValueError("Expected torch.Tensor input")
        
        if len(image.shape) != 4 or image.shape[-1] != 3:
            raise ValueError("Expected video input with shape [frames, height, width, 3]")
        
        original_method = self.predictor.add_new_points_or_box
        
        def patched_method(*args, **kwargs):
            print("\nDebug: add_new_points_or_box called")
            print(f"Debug: kwargs keys = {kwargs.keys()}")
            
            try:
                if 'box' in kwargs:
                    box = kwargs['box']
                    print(f"Debug: box device = {box.device}")
                    print(f"Debug: box shape = {box.shape}")
                    print(f"Debug: box dtype = {box.dtype}")
                    
                    points = torch.zeros((0, 2), dtype=box.dtype, device=box.device)
                    labels = torch.zeros((0,), dtype=torch.int64, device=box.device)
                    
                    kwargs['points'] = points
                    kwargs['labels'] = labels
                
                for attr_name in dir(self.predictor):
                    attr = getattr(self.predictor, attr_name)
                    if isinstance(attr, torch.Tensor):
                        print(f"Debug: predictor.{attr_name} device = {attr.device}")
                        if attr.device.type != 'cuda':
                            setattr(self.predictor, attr_name, attr.to('cuda'))
                
                result = original_method(*args, **kwargs)
                return result
            except Exception as e:
                print(f"Error in patched method: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                raise
        
        self.predictor.add_new_points_or_box = patched_method
        
        # Обрезаем видео с start_frame
        image = image[start_frame:]
        num_frames = len(image)
        print(f"Processing video from frame {start_frame}, total frames to process: {num_frames}")
        
        # Получаем размеры уже обрезанного видео
        num_frames, h, w, c = image.shape
        
        # Ресайз если нужно
        max_side = max(h, w)
        if max_side > resolution:
            scale = resolution / max_side
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            print(f"Resizing image from {h}x{w} to {new_h}x{new_w}")
            print(f"Aspect ratio: {w/h:.2f} (original) -> {new_w/new_h:.2f} (resized)")
            
            image = F.interpolate(
                image.permute(0, 3, 1, 2),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            h, w = new_h, new_w
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            frames_list = []
            for i in range(num_frames):
                frame = image[i]
                frame_np = frame.cpu().numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                frames_list.append(frame_np)
            
            image_np = np.stack(frames_list, axis=0)
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for i, frame in enumerate(image_np):
                        frame_number = start_frame + i
                        Image.fromarray(frame).save(os.path.join(temp_dir, f"{frame_number:05d}.jpg"), quality=95)
                    
                    inference_state = self.predictor.init_state(
                        video_path=temp_dir,
                        offload_video_to_cpu=False
                    )
                    
                    if box is not None:
                        box_tensor = torch.tensor([[box[0], box[1], box[0] + box[2], box[1] + box[3]]], 
                                                dtype=torch.float16, 
                                                device=self.device)
                        print(f"Box tensor device: {box_tensor.device}")
                        print(f"Box tensor shape: {box_tensor.shape}")
                        print(f"Box tensor: {box_tensor}")
                        
                        _, obj_ids, masks = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=0,
                            obj_id=1,
                            box=box_tensor,
                            normalize_coords=True
                        )

                    if points is not None and labels is not None:
                        points_tensor = torch.tensor(points, dtype=torch.float16, device='cuda')
                        labels_tensor = torch.tensor(labels, dtype=torch.int64, device='cuda')
                        print(f"Points tensor device: {points_tensor.device}")
                        print(f"Points tensor shape: {points_tensor.shape}")
                        print(f"Labels tensor device: {labels_tensor.device}")
                        print(f"Labels tensor shape: {labels_tensor.shape}")
                        
                        _, obj_ids, masks = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=0,
                            obj_id=1,
                            points=points_tensor,
                            labels=labels_tensor,
                            normalize_coords=True
                        )
                    
                    all_masks = []
                    if True:  # Всегда добавляем первую маску
                        mask = (masks > iou_threshold).float()
                        all_masks.append(mask)
                    
                    for current_frame_idx, current_obj_ids, current_masks in self.predictor.propagate_in_video(inference_state):
                        mask = (current_masks > iou_threshold).float()
                        
                        while len(all_masks) < current_frame_idx + 1:
                            all_masks.append(None)
                        all_masks[current_frame_idx] = mask
                        
                        print(f"Processed frame {current_frame_idx + start_frame}/{num_frames + start_frame - 1}")
                    
                    if None in all_masks:
                        raise ValueError("Some frames were not processed during propagation")
                    
                    sequence_masks = torch.stack(all_masks, dim=0)
                    
                    del inference_state
                    cleanup_memory()
                    
                    return (sequence_masks,)
                    
            except Exception as e:
                print(f"Error during segmentation: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                raise