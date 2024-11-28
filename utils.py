import torch
import numpy as np
from loguru import logger
import gc
import time

def setup_logging():
    """Настройка логирования"""
    logger.add("samurai_debug.log", 
               rotation="100 MB", 
               level="DEBUG",
               format="{time} {level} {message}")

def get_device_info():
    """Получение информации об устройстве"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        return {
            "name": device.name,
            "total_memory": device.total_memory / 1024**2,
            "major": device.major,
            "minor": device.minor
        }
    return {"name": "CPU", "total_memory": 0, "major": 0, "minor": 0}

def optimize_tensor(tensor):
    """Оптимизация тензора для экономии памяти"""
    if tensor is None:
        return None
    
    if isinstance(tensor, torch.Tensor):
        # Используем half precision где возможно
        if tensor.dtype in [torch.float32, torch.float64]:
            tensor = tensor.half()
        return tensor.contiguous()
    return tensor

def batch_process(items, batch_size, process_fn):
    """Обработка элементов батчами"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
        cleanup_memory()
    return results