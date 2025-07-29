# GCN-ConvNeXt: Cotton Disease Classification

This repository contains code and models for classifying cotton plant diseases using a hybrid approach that combines ConvNeXt (a modern convolutional neural network) with Graph Convolutional Networks (GCN) on superpixel-based graph representations.

## Features
- **ConvNeXt Backbone:** Extracts deep features from cotton leaf images.
- **Superpixel Graph Construction:** Converts images into superpixel graphs for spatial reasoning.
- **GCN Layers:** Learns relationships between superpixels for improved disease classification.
- **Jupyter Notebook:** End-to-end workflow for training, evaluation, and prediction.

## Requirements
- Python 3.8+
- PyTorch
- torch-geometric
- torchvision
- scikit-image
- tensorflow (for Keras ImageDataGenerator)
- matplotlib
- PIL (Pillow)

Install dependencies with:
```sh
pip install torch torchvision torch-geometric scikit-image tensorflow matplotlib pillow
```

## Usage
1. **Prepare Dataset:**
   - Organize your dataset in the following structure:
     ```
     Cotton_Dataset/
       train/
         class1/
         class2/
         ...
       val/
         class1/
         class2/
         ...
     ```
2. **Run the Notebook:**
   - Open `Convnext_GCN.ipynb` in Jupyter or VS Code.
   - Follow the cells to train or evaluate the model.

3. **Model Weights:**
   - The file `convnext_gcn_model_full.pth` is **not included** due to GitHub file size limits. Train your own or contact the author for access.

## Notes
- Checkpoints and large model files are excluded from the repository via `.gitignore`.
- For best results, use a GPU-enabled environment.

## Citation
If you use this code, please cite the repository or acknowledge the authors.

## License
MIT License
