# SAMURAI Nodes for ComfyUI

ComfyUI nodes for video object segmentation using [SAMURAI](https://github.com/yangchris11/samurai) model.

## Installation

1. Follow the [SAMURAI installation guide](https://github.com/yangchris11/samurai) to install the base model
2. Clone this repository into your ComfyUI custom nodes directory:

bash

cd ComfyUI/custom_nodes

git clone https://github.com/your-username/samurai_nodes.git

3. Copy the SAMURAI installation into `ComfyUI/custom_nodes/samurai_nodes/samurai`
4. Download model weights as described in [SAMURAI guide](https://github.com/yangchris11/samurai)

## Additional Dependencies

Most dependencies are included with SAMURAI installation. Additional required packages:

bash
pip install hydra-core omegaconf loguru

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