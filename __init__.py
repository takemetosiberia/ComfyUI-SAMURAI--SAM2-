import os
import subprocess
import sys
from pathlib import Path
from .utils import setup_logging, get_device_info
from loguru import logger

# Настройка логирования
setup_logging()

# Вывод информации об устройстве
device_info = get_device_info()
logger.info(f"Running on device: {device_info['name']}")
logger.info(f"Available memory: {device_info['total_memory']:.2f} MB")

# Установка зависимостей
def install_requirements():
    requirements_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])

try:
    import loguru
except ImportError:
    print("Installing required packages...")
    install_requirements()
    import loguru

# Добавляем путь к SAM2 в PYTHONPATH
CURRENT_DIR = Path(__file__).parent
SAM2_PATH = CURRENT_DIR / "samurai" / "sam2" / "sam2"
if str(SAM2_PATH) not in sys.path:
    sys.path.insert(0, str(SAM2_PATH))

from .samurai_node import SAMURAIBoxInputNode, SAMURAIPointsInputNode, SAMURAIRefineNode

NODE_CLASS_MAPPINGS = {
    "SAMURAIBoxInputNode": SAMURAIBoxInputNode,
    "SAMURAIPointsInputNode": SAMURAIPointsInputNode,
    "SAMURAIRefineNode": SAMURAIRefineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMURAIBoxInputNode": "SAMURAI Box Input",
    "SAMURAIPointsInputNode": "SAMURAI Points Input",
    "SAMURAIRefineNode": "SAMURAI Refine"
}