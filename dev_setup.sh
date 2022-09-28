cd ..
git clone https://github.com/qway/nerfmeshes.git

cd nerfmeshes-manage
pip install -r requirements.txt
pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu113
conda install pytorch3d -c pytorch3d

