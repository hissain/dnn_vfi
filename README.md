# Video Frame Interpolation (VFI) - Demo Using DNN

This repository contains a machine learning demo for video frame interpolation (VFI) using three different models: UNet, RIFE, and Mamba. The goal is to predict an intermediate frame (2nd frame) given the 1st and 3rd frames as input. The project is under the MIT license.

## 📂 Project Structure
```
.
├── LICENSE                # MIT License
├── README.md              # Project Documentation
├── __pycache__/           # Compiled Python files
├── input/                 # Sample videos
│   ├── enjoy.mp4
│   ├── glob.mp4
│   └── motion.mp4
├── mamba/                 # Mamba-based VFI
│   ├── vfi.ipynb
│   └── vfi_model.py
├── requirements.txt       # Required dependencies
├── rife/                  # RIFE-based VFI
│   └── RIFE.ipynb
└── unet/                  # UNet-based VFI
    ├── Unet2d.ipynb
    └── Unet3d.ipynb
```

## 🚀 Models Implemented

__UNet:__ Convolutional neural network (CNN) based architecture for frame interpolation.

__RIFE:__ Real-time Intermediate Flow Estimation model.

__Mamba:__ A sequence modeling architecture applied to frame interpolation.

## 📌 Features

* Sample videos included for testing.

* Frame extraction and dataset preparation.

* Training notebooks for different models.

## 🔧 Installation

Clone the repository and install dependencies:

```
git clone https://github.com/hissain/dnn_vfi.git
cd dnn_vfi
pip install -r requirements.txt
```

## 📊 Usage

Run the corresponding Jupyter notebooks inside the mamba/, rife/, or unet/ directories to train and evaluate the models.

jupyter notebook

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the project!

## 📬 Contact

For any inquiries, reach out to [hissain.khan@gmail.com] or create an issue in the repository.
