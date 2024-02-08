apt update
apt upgrade -y
apt dist-upgrade -y
apt install gcc-riscv64-linux-gnu nano -y
mkdir -p /tmp/sopt
mount -t tmpfs -o size=4096M tmpfs /tmp/sopt