# Hierarchical-world-model

Deep Hierarchical Planning from Pixels
======================================

Official implementation of the [Director][project] algorithm in TensorFlow 2.

[project]: https://danijar.com/director/

![Director Internal Goals](https://github.com/danijar/director/raw/main/media/header.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2022director,
  title={Deep Hierarchical Planning from Pixels},
  author={Hafner, Danijar and Lee, Kuang-Huei and Fischer, Ian and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

How does Director work?
-----------------------

Director is a practical and robust algorithm for hierarchical reinforcement
learning. To solve long horizon tasks end-to-end from sparse rewards, Director
learns to break down tasks into internal subgoals. Its manager policy selects
subgoals that trade off exploratory and extrinsic value, its worker policy
learns to achieve the goals through low-level actions. Both policies are
trained from imagined trajectories predicted by a learned world model. To
support the manager in choosing realistic goals, a goal autoencoder compresses
and quantizes previously encountered representations. The manager chooses its
goals in this compact space. All components are trained concurrently.

![Director Method Diagram](https://github.com/danijar/director/raw/main/media/method.png)

For more information:

- [Google AI Blog](https://ai.googleblog.com/2022/07/deep-hierarchical-planning-from-pixels.html)
- [Project website](https://danijar.com/project/director/)
- [Research paper](https://arxiv.org/pdf/2206.04114.pdf)

Running the Agent
-----------------

Dockerfile
```sh
docker run -it --rm --gpus all tensorflow/tensorflow:2.13.0-gpu nvidia-smi

docker build -f agents/director/Dockerfile -t img . && \
docker run -it --rm --gpus all -v ~/logdir:/logdir img \
    sh /embodied/scripts/xvfb_run.sh python3 agents/director/train.py \
    --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
    --configs dmc_vision --task dmc_walker_walk
```

Either use `embodied/agents/director/Dockerfile` or follow the manual instructions below.

Install dependencies:

omnicampus環境の場合

```sh
apt update
apt install sudo
sudo apt update
sudo apt install -y python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
```

CUDAを入れる前に(一応)
```sh
sudo apt-get update
sudo apt-get install -y dialog
sudo apt-get install -y kmod
```

```sh
git clone https://github.com/s1f10220227/Hierarchical-world-model
```

Ubuntuのバージョンを確認
```sh
cat /etc/os-release
arch
uname -r
gcc --version
cat /proc/version
```

バージョンを確認
```sh
nvidia-smi
nvcc --version
dpkg -l | grep cudnn
```

```sh
ls /usr/local/ -la
ls /usr/local/
sudo update-alternatives --config cuda
```

依存関係を確認
https://www.tensorflow.org/install/source#gpu
https://keras.io/getting_started/
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html


```sh
sudo apt-get install linux-headers-$(uname -r)
```

CUDAのインストール
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
sudo apt-get -y install cuda-toolkit-11-8
```

注意: driverを消さない場合はtoolkitをつける
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages

環境変数を設定
```sh
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
# CUDA バージョンを確認
```sh
nvidia-smi
nvcc --version
```

CUDAを切り替えたい場合
```sh
sudo update-alternatives --config cuda
```

cuDNN のインストール
```sh
dpkg -l | grep cudnn
```
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn-cuda-11
```

# 環境変数を再ロード
source ~/.bashrc

# cuDNNのバージョンを確認
```sh
dpkg -l | grep cudnn
```
```sh
sudo apt update
sudo apt install -y \
  ffmpeg git python3-pip vim wget unrar xvfb \
  libegl1-mesa libopengl0 libosmesa6 libgl1-mesa-glx libglfw3
```
# Atari環境のセットアップスクリプトを実行
```sh
cd Hierarchical-world-model/embodied/
sh scripts/install-atari.sh
```
```sh
cd ..
pip install --no-cache-dir -r requirements.txt
```
MuJoCo GL設定
```sh
export MUJOCO_GL=egl
```
TensorFlowとXLA設定
```sh
export TF_FUNCTION_JIT_COMPILE_DEFAULT=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

RuntimeErrorのとき
tensorflowとkerasのバージョンの依存関係を確認

Train agent:

embodiedにいるとき
```sh
sh scripts/xvfb_run.sh python3 agents/director/train.py   
    --logdir "/logdir/$(date +%Y%m%d-%H%M%S)"   
    --configs dmc_vision 
    --task dmc_walker_walk

```

See `agents/director/configs.yaml` for available flags and
`embodied/envs/__init__.py` for available envs.

Using the Tasks
---------------

The HRL environments are in `embodied/envs/pinpad.py` and
`embodied/envs/loconav.py`.
