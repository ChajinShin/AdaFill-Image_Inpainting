# AdaFill
AdaFill: Test-time Adaptation for Out-of-Distributed Image Inpainting



## Prerequisites
We used following components:
- Python3.7+
- PyTorch 1.7.1
- torchvision 0.8.2
- scikit-image
- PyYAML
- PIL
- numpy
- opencv-python






## Pretrain the model
Modify 'task' option in *options.yml* file to 'Pretraining'
  -  task: 'Pretraining'


Specify dataset location in 'dataset_dir' option in *options.yml* file
  -  dataset_dir: '/home/Dataset/places365'


run *main.py*
  -  python main.py







## AdaFill for test time adapation
First, you need to make file_list using *make_flist.py* file.
Modify following two folder directories in *make_flist.py* file
  -  IMAGE_FOLDER = r'./home/Dataset/AdaFill/image'
  -  MASK_FOLDER = r'./home/Dataset/AdaFill/mask'


Get *flist.txt*
  -  python make_flist.py


Then modify 'task' option in *options.yml* file to 'AdaFill'
  -  task: 'AdaFill'


You can get results.
  -  python main.py

