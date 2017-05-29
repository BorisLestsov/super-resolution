import torch
import os
import skimage.io
import skimage.transform
from PIL import Image
import numpy
import shutil

num_upscales = 1
max_upscale = 2048

test_data_dir = '../super-res-test-div2'
target_dir = test_data_dir + '-preds'

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
    
os.mkdir(target_dir)

files = os.listdir(test_data_dir)
files = sorted(files)

model = torch.load('model_dump.pt').cuda()

scale = 2 ** num_upscales
        

for file_name in files:
    if file_name[0] != '.':
        
        full_path = test_data_dir + '/' + file_name
        print(full_path)
        
        target_path = target_dir + '/' + file_name
        img = Image.open(full_path).convert('RGB')
        img = numpy.array(img)
        res = numpy.zeros([img.shape[0] * scale, img.shape[1] * scale, 3])
        
        img = img.swapaxes(1, 2).swapaxes(0, 1)
        img = img.reshape((1,) + img.shape)
        img = img / 256.0
        
        frac_step = int(max_upscale / scale)
        
        
        for x_index in range(0, img.shape[2], frac_step):
            for y_index in range(0, img.shape[3], frac_step):
                img_frac = img[:, :, x_index:x_index + frac_step, y_index:y_index + frac_step]    
                
        
                img_frac = torch.Tensor(img_frac)
                img_frac = torch.autograd.Variable(img_frac).cuda()
                
                #print(img_frac.size(), '<- image fraction size')
                preds = model(img_frac, num_upscales)
        
                res_frac = preds[-1][0].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
               
                #print(frac_step)
                #print(res_frac.shape, '<- res frac shape')
                #print(res[x_index * scale:(x_index + frac_step) * scale,
                #    y_index * scale:(y_index + frac_step) * scale, :].shape, '<- to write shape')
                res[x_index * scale:(x_index + frac_step) * scale,
                    y_index * scale:(y_index + frac_step) * scale, :] = res_frac
        skimage.io.imsave(target_path, res.clip(0.0, 1.0))
        
        del(img)
        del(res)
        del(preds)
        