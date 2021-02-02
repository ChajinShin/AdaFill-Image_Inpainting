import os


IMAGE_FOLDER = r'./ablation/image'
MASK_FOLDER = r'./ablation/ff'
RESULT_FOLDER = r'./inpaint_exp/results'

img_list = sorted(os.listdir(IMAGE_FOLDER))
mask_list = sorted(os.listdir(MASK_FOLDER))

with open('flist.txt', 'w') as f:
    for idx, (img, mask) in enumerate(zip(img_list, mask_list)):
        img_loc = os.path.join(IMAGE_FOLDER, img)
        mask_loc = os.path.join(MASK_FOLDER, mask)
        str = '{},{},{}/result_{:04d}.jpg'.format(img_loc, mask_loc, RESULT_FOLDER, idx)
        f.write(str)
        f.write('\n')


