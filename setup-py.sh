sudo apt update
sudo apt install -y python3-pip
sudo apt install -y python3.12-venv
python3 -m venv ml-env
source ml-env/bin/activate
pip install torch torchvision torchaudio
pip install matplotlib seaborn
