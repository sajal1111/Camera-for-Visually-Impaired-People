# InternVL2 Live Webcam Captioning (CUDA)

Real-time webcam image captioning using InternVL2-1B with CUDA acceleration.

## Features
- Live webcam captioning
- InternVL2-1B vision-language model
- FP16 CUDA inference (RTX 3050 tested)
- Similarity-based caption filtering
- OpenCV real-time overlay

## Demo
![Live Demo](demo/demo.gif)

## Requirements
- Python 3.10
- NVIDIA GPU with CUDA

## Run
```bash
pip install -r requirements.txt
python live_internvl_webcam.py
