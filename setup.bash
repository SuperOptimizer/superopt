apt update
apt upgrade -y
apt dist-upgrade -y
apt install gcc-riscv64-linux-gnu nano zsh sentencepiece -y
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
sed -i -e 's/robbyrussell/gentoo/g' ~/.zshrc
source ~/.zshrc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
mkdir -p /tmp/sopt
mount -t tmpfs -o size=4096M tmpfs /tmp/sopt
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f
/home/user/miniconda3/bin/conda config --set auto_activate_base true
