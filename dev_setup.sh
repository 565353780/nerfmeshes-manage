cd ..
git clone https://github.com/qway/nerfmeshes.git

cd nerfmeshes-manage
pip install -r requirements.txt
pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html

