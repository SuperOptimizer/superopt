apt update
apt upgrade -y
apt dist-upgrade -y
apt install gcc-riscv64-linux-gnu nano sentencepiece -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
mkdir -p /tmp/sopt
mount -t tmpfs -o size=4096M tmpfs /tmp/sopt