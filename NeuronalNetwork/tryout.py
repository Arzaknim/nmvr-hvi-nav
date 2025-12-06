import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPUs detected:", tf.config.list_physical_devices('GPU'))

#  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6