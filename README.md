

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

