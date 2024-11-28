# SAMURAI Nodes for ComfyUI

ComfyUI nodes for video object segmentation using [SAMURAI](https://github.com/yangchris11/samurai) model.

## Installation

> **Note:** It is recommended to use Conda environment for installation and running the nodes.
> Make sure to use the same Conda environment for both ComfyUI and SAMURAI installation!
> ## Requirements
- NVIDIA GPU with CUDA support
- Python 3.10 or higher
- ComfyUI
- Conda (recommended) or pip

1. Follow the [SAMURAI installation guide](https://github.com/yangchris11/samurai) to install the base model

2. Clone this repository into your ComfyUI custom nodes directory:

```
cd ComfyUI/custom_nodes

git clone https://github.com/takemetosiberia/ComfyUI-SAMURAI--SAM2-.git samurai_nodes
```

3. Copy the SAMURAI installation folder into `ComfyUI/custom_nodes/samurai_nodes/`

4. Download model weights as described in [SAMURAI guide](https://github.com/yangchris11/samurai)

5. ## Project Structure

After installation, your directory structure should look like this:

```
ComfyUI/
└── custom_nodes/
    └── samurai_nodes/
        ├── samurai/     # SAMURAI model installation
        ├── init.py      # Module initialization
        ├── samurai_node.py
        └── utils.py
```

## Additional Dependencies

Most dependencies are included with SAMURAI installation. Additional required packages:

```
pip install hydra-core omegaconf loguru
```

## Usage

The workflow consists of three main nodes:

### SAMURAI Box Input
Allows selecting a region of interest (box) in the first frame of a video sequence. 
- Input: video frames
- Output: box coordinates and start frame number

### SAMURAI Points Input
Enables point-based object selection in the first frame.
- Input: video frames
- Output: point coordinates, labels, and start frame number

### SAMURAI Refine
Performs video object segmentation using selected area.
- Input: video frames, box/points from input nodes
- Output: segmentation masks

## Example Workflow

1. Connect Load Video to SAMURAI Box/Points Input
2. Draw box or place points around object of interest
3. Connect to SAMURAI Refine
4. Convert masks to images and save/combine as needed

For more examples and details, see [SAMURAI documentation](https://github.com/yangchris11/samurai).

## Troubleshooting

If you encounter any issues:
1. Make sure you're using the correct Conda environment
2. Verify that all dependencies are installed in your Conda environment
3. Check if SAMURAI model is properly installed in the `samurai/` directory

For CUDA-related issues, ensure your Conda environment has the correct PyTorch version with CUDA support.
