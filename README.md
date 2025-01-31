# Hierarchical-world-model

Deep Hierarchical Planning from Pixels
======================================

Official implementation of the [Director][project] algorithm in TensorFlow 2.

[project]: https://danijar.com/director/

![Director Internal Goals](https://github.com/danijar/director/raw/main/media/header.gif)



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

Dockerfileの場合
```sh
docker run -it --rm --gpus all tensorflow/tensorflow:2.13.0-gpu nvidia-smi

docker build -f agents/director/Dockerfile -t img . && \
docker run -it --rm --gpus all -v ~/logdir:/logdir img \
    sh /embodied/scripts/xvfb_run.sh python3 agents/director/train.py \
    --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
    --configs dmc_vision --task dmc_walker_walk
```

Install dependencies:

Dockerを使わない場合

今回は(omnicampus環境)

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
sudo apt-get -y install cuda-toolkit-11-8
```

注意: driverを消さない場合はtoolkitをつける  
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages

環境変数を設定
```sh
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}
```
CUDA バージョンを確認
```sh
nvidia-smi
nvcc --version
```

CUDAを切り替えたい場合
```sh
sudo update-alternatives --config cuda
```

cuDNN(8.6.0.)のインストール

https://developer.nvidia.com/rdp/cudnn-archive

cuDNNのバージョンを確認
```sh
dpkg -l | grep cudnn
```
```sh
sudo apt remove --purge 'libcudnn9-*'
sudo apt autoremove
sudo apt clean
```
パスや鍵名を修正してください。
```sh
sudo dpkg -i <path/to/cudnn.deb>
sudo cp /var/cudnn-local-repo-*/<key-file-name> /usr/share/keyrings/
sudo apt update
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8
```
cuDNNのバージョンを確認
```sh
dpkg -l | grep cudnn
```

必要なパッケージのインストール
```sh
sudo apt update
sudo apt install -y \
  ffmpeg git python3-pip vim wget unrar xvfb \
  libegl1-mesa libopengl0 libosmesa6 libgl1-mesa-glx libglfw3
```
Atari環境のセットアップスクリプトを実行
```sh
cd Hierarchical-world-model/embodied/
sh scripts/install-atari.sh
```
パッケージのインストール
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
環境変数を再ロード
```sh
source ~/.bashrc
```


Train agent:

embodiedにいるとき
embodied/agents/director/configs.yamlを見てで環境とタスクを選ぶ
```sh
sh scripts/xvfb_run.sh \
  python3 agents/director/train.py \
  --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
  --configs dmc_vision \
  --task dmc_walker_walk
```
結果の確認
```sh
tensorboard --logdir ~/logdir
```


RuntimeErrorのとき
tensorflowとkerasのバージョンの依存関係を確認

See `agents/director/configs.yaml` for available flags and
`embodied/envs/__init__.py` for available envs.

Using the Tasks
---------------

The HRL environments are in `embodied/envs/pinpad.py` and
`embodied/envs/loconav.py`.
