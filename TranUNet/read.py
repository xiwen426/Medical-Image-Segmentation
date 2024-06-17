# test datasets content
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 加载 NIfTI 文件
file_path = '/data2/wyx/medical/amos22/labelsTr/amos_0001.nii.gz'
img = nib.load(file_path)

# 获取数据
data = img.get_fdata() # (768, 768, 90)

# 打印数据形状
print(f'Data shape: {data.shape}')

# 显示中间切片
slice_index = data.shape[2] // 2
plt.imshow(data[:, :, slice_index], cmap='gray')
plt.title(f'Slice {slice_index}')
plt.savefig('slice.png')
