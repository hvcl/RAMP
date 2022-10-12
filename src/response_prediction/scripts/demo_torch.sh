docker run --gpus '"device=0"' --rm -v /home:/home --shm-size=256GB -ti kglee/pytorch:1.12.0 python src/response_prediction/scripts/demo_torch.py
