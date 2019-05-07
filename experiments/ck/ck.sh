#!/usr/bin/env bash
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python ${SOURCE_DIR}/ck.py \
-arch vgg16 -ep 32 -bs 30 -lr 0.01 -lrs 8 -nc 7 -epl 600 \
-od /media/shehabk/D_DRIVE/codes/code_practice/pytorch_ck/outputs_n \
-imr /media/shehabk/D_DRIVE/codes/code_practice/pytorch_ck/data/cropped_256 \
-imtr /media/shehabk/D_DRIVE/codes/code_practice/pytorch_ck/image_lists/set1/train.txt \
-imvl /media/shehabk/D_DRIVE/codes/code_practice/pytorch_ck/image_lists/set1/val.txt \
-imts /media/shehabk/D_DRIVE/codes/code_practice/pytorch_ck/image_lists/set1/test.txt \
-m train