docker run --gpus '"device=0"' --rm -v /home:/home -ti ramp/tf:2.6.0 python src/response_prediction/scripts/demo_tf.py
