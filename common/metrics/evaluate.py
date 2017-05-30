#-*- coding: utf8 -*-
import os, sys, cv2, math
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from sys import argv


def compare_two_dir(test, source):
    if not os.path.exists(test) or not os.path.exists(source):
        raise(ValueError, 'The dirs don\'t exist')

    psnr_sum, ssim_sum, count = 0, 0, 0

    for fname in sorted(os.listdir(source)):
        print('Comparing %s' % fname)
        test_img_path = os.path.join(test, fname)
        source_img_path = os.path.join(source, fname)
        if not os.path.exists(test_img_path):
            raise(ValueError, 'Filename %s doesn\'t exist' % (test_img_path))
        source_img = cv2.imread(source_img_path)
        test_img = cv2.imread(test_img_path)

        #TEST
        #test_img = cv2.resize(test_img, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        #

        psnr = compare_psnr(source_img, test_img)
        if math.isinf(psnr):
            psnr = 1.0
        ssim = compare_ssim(cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        print(fname, psnr, ssim)
        psnr_sum += psnr
        ssim_sum += ssim
        count += 1
        #if count > 10:
        #    break

    print('PSNR=%.5f, SSIM=%.5f' % (psnr_sum/float(count), ssim_sum/float(count)))



def main():
    compare_two_dir(argv[1],  argv[2])


if __name__ == '__main__':
    sys.exit(main())

