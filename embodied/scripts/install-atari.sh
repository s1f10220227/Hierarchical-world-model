#!/bin/sh
set -eu

apt-get update
apt-get install -y wget
apt-get install -y unrar
apt-get clean

pip3 install --no-cache-dir gym
pip3 install --no-cache-dir atari-py
pip3 install --no-cache-dir opencv-python

wget -L -nv http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar
python3 -m atari_py.import_roms ROMS
rm -rf Roms.rar ROMS.zip ROMS
