#!/bin/sh
echo '解压tar包...'
tar -zxvf docker-19.03.8.tgz
echo '将docker目录移到/usr/bin目录下...'
cp docker/* /usr/bin/
echo '将docker.service 移到/etc/systemd/system/ 目录...'
cp docker.service /usr/lib/systemd/system/
echo '添加文件权限...'
chmod +x /usr/lib/systemd/system/docker.service
echo '重新加载配置文件...'
systemctl daemon-reload
echo '将daemon.json 移到/etc/docker/ 目录...'
cp daemon.json /etc/docker/
echo '重新加载配置文件...'
systemctl daemon-reload
echo '启动docker...'
systemctl start docker
echo '设置开机自启...'
systemctl enable docker.service
echo 'docker安装成功...'
docker -v
