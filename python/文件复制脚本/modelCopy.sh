#!/bin/sh
echo '执行python复制文件脚本========'
sudo conda activate ai
sudo python3 model.py

echo '对文件进行压缩====='
zip modelfile.zip  modelfile/


