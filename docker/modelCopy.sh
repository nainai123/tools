#!/bin/sh

echo "请按Enter确认删除/data2/gztpai/modelfile/code model 文件夹内文件"
read

rm -rf /data2/gztpai/modelfile/code/*
rm -rf /data2/gztpai/modelfile/model/*
echo "文件删除完毕，请根据提示输入。。。"


echo '正常执行python复制文件脚本========'
sudo conda activate ai
sudo python3 model.py
echo '复制完毕正在对文件进行压缩，请稍等========'

zip modelfile.zip  modelfile/
echo '压缩完毕请提取文件，感谢使用====='

