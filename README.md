

## ⚙️ Installation

**Create environment:**

```bash

# 1. Install requirements
pip install -r requirements.txt

# 3. Install ffmpeg
sudo apt-get update  
sudo apt-get install ffmpeg -y

# For Windows
Follow the instructions [ffmpeg-windows](https://phoenixnap.com/kb/ffmpeg-windows)

# 4. Install MultiScaleDeformableAttention (only availabe in CUDA environmets)
cd src/utils/dependencies/XPose/models/UniPose/ops
python setup.py build install
cd - # equal to cd ../../../../../../../
```

## 🎒 Prepare model checkpoints

Run the `download_model.py` to prepare the following `pretrained_weights` directory.

The final `pretrained_weights` directory should look like this:

```text
./pretrained_weights/
├── insightface                                                                                                                                                 
│   └── models                                                                                                                                                  
│       └── buffalo_l                                                                                                                                           
│           ├── 2d106det.onnx                                                                                                                                   
│           └── det_10g.onnx   
├── JoyVASA
│   ├── motion_generator
│   │   └── iter_0020000.pt
│   └── motion_template
│       └── motion_template.pkl
├── liveportrait
│   ├── base_models
│   │   ├── appearance_feature_extractor.pth
│   │   ├── motion_extractor.pth
│   │   ├── spade_generator.pth
│   │   └── warping_module.pth
│   ├── landmark.onnx
│   └── retargeting_models
│       └── stitching_retargeting_module.pth
├── liveportrait_animals
│   ├── base_models
│   │   ├── appearance_feature_extractor.pth
│   │   ├── motion_extractor.pth
│   │   ├── spade_generator.pth
│   │   └── warping_module.pth
│   ├── retargeting_models
│   │   └── stitching_retargeting_module.pth
│   └── xpose.pth
├── TencentGameMate:chinese-hubert-base
│   ├── chinese-hubert-base-fairseq-ckpt.pt
│   ├── config.json
│   ├── gitattributes
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin
│   └── README.md
└── wav2vec2-base-960h               
    ├── config.json                  
    ├── feature_extractor_config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    ├── pytorch_model.bin
    ├── README.md
    ├── special_tokens_map.json
    ├── tf_model.h5
    ├── tokenizer_config.json
    └── vocab.json
```

> [!NOTE]
> The folder `TencentGameMate:chinese-hubert-base` in Windows should be renamed `chinese-hubert-base`.

## 🚀 Inference

Animal:

```python
python inference.py -r assets/examples/imgs/joyvasa_001.png -a assets/examples/audios/joyvasa_001.wav --animation_mode animal --cfg_scale 2.0
```
Or

```python
python test_animal.py
```


# CUDA and PyTorch Installation Guide for Ubuntu

This guide will help you install the NVIDIA drivers, CUDA toolkit, CuDNN, and PyTorch on an Ubuntu system. Follow these steps carefully to set up your environment for GPU-accelerated machine learning.

---

## 1. Upgrade your Ubuntu system

First, make sure your Ubuntu system is up to date. This ensures you have the latest software and security updates installed.

```bash
sudo apt update
sudo apt upgrade
```

---

## 2. List the recommended NVIDIA drivers

We need to list the recommended NVIDIA drivers for your system. To do this, install the `ubuntu-drivers-common` package and run the command to check for the best driver for your GPU.

```bash
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers devices
```

This command will show the recommended NVIDIA driver for your hardware.

---

## 3. Install the recommended NVIDIA driver

Install the NVIDIA driver recommended for your GPU. For example, if `nvidia-driver-550` is the suggested driver, install it by running:

```bash
sudo apt install nvidia-driver-550
```

---

## 4. Reboot your system

After installing the driver, reboot your system to ensure the driver is properly loaded.

```bash
sudo reboot now
```

---

## 5. Check the driver installation

Once your system reboots, check if the NVIDIA driver is properly installed by running the following command:

```bash
nvidia-smi
```

This will display the current state of the GPU and confirm that the driver is working correctly.

---

## 6. Install GCC

The GCC compiler is required for installing the CUDA toolkit. To ensure that GCC is installed, use the following command:

```bash
sudo apt install gcc
```

To verify if GCC has been installed correctly, run:

```bash
gcc -v
```

---

## 7. Install CUDA toolkit on Ubuntu

Download the CUDA toolkit from the official NVIDIA website using this [link](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network).

Then install it with the following commands:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

---

## 8. Reboot your system

Now that the CUDA toolkit has been installed, reboot the system to load the right modules required by CUDA:

```bash
sudo reboot now
```

---

## 9. Environment setup

The CUDA toolkit is now installed, and a few manual actions must be executed to complete the setup. We will now proceed to update the environment variables as recommended by the NVIDIA documentation.

Add the following lines to your `.bashrc` file. Use the command `nano ~/.bashrc` to open the file and paste these lines at the end of the file:

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Save the file using `Ctrl+x`, `y`, and then press `ENTER`.

Now, reload the `.bashrc` file:

```bash
. ~/.bashrc
```

---

## 10. Test the CUDA toolkit

Now that the environment has been set up, you can test the CUDA toolkit and execute the `nvcc` (CUDA compiler). Run:

```bash
nvcc -V
```

It should display the installed version of `nvcc`.

---

## 11. Install CUDNN

Download CuDNN from the following [link](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local).

To install CuDNN, run these commands:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

---

## 12. Install PyTorch

Finally, install PyTorch with the following command:

```bash
pip3 install torch torchvision torchaudio
```

