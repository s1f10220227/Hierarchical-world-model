# 1. Test setup:
# docker run -it --rm --gpus all tensorflow/tensorflow:2.13.0-gpu nvidia-smi
#
# 2. Start training:
# docker build -f agents/director/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#    sh /embodied/scripts/xvfb_run.sh python3 agents/director/train.py \
#    --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#    --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM tensorflow/tensorflow:2.13.0-gpu

# Disable pip version check and ensure output is unbuffered
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1

# Clean unnecessary sources and install system dependencies
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y \
  apt ffmpeg git python3-pip vim wget unrar xvfb \
  libegl1-mesa libopengl0 libosmesa6 libgl1-mesa-glx libglfw3 \
  && apt-get clean

# Install Python dependencies
RUN pip3 install --no-cache-dir \
  crafter \
  robodesk \
  dm_control \
  dm-sonnet \
  tensorflow_probability \
  numpy cloudpickle ruamel.yaml rich \
  PyOpenGL

# Copy and execute the Atari setup script
COPY scripts scripts
RUN sh scripts/install-atari.sh

# Set MuJoCo environment variable to use egl
ENV MUJOCO_GL egl

# Configure TensorFlow and XLA settings
ENV TF_FUNCTION_JIT_COMPILE_DEFAULT 1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Embodied specific setup
COPY . /embodied
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied

# Set working directory
WORKDIR /embodied

# Default command to start training
CMD [ \
  "python3", "agents/director/train.py", \
  "--logdir=/logdir/$(date +%Y%m%d-%H%M%S)", \
  "--configs=dmc_vision", \
  "--task=dmc_walker_walk" \
]
