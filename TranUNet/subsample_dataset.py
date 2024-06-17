import argparse
import os
import numpy as np
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import h5py

def main(args):
    root, output, ds_res = args.root, args.output, args.ds_res
    output = output + f'{str(ds_res)}'
    if not os.path.exists(output):
        os.makedirs(output)
    ct_slices = os.listdir(root)
    p_bar = tqdm(ct_slices)
    for ct_slice in p_bar:
        # print(os.path.join(root, ct_slice))
        # slice_data = np.load(os.path.join(root, ct_slice))
        # image_data, label_data = slice_data['image'], slice_data['label']
        if ct_slice == '224':
            continue
        file_path = os.path.join(root, ct_slice)
        # print(file_path)
        data = h5py.File(file_path)
        image_data, label_data = data['image'][:], data['label'][:]
        # print(image_data.shape, label_data.shape)
        _, h, w = image_data.shape
        # Calculate zoom factor
        if len(image_data.shape) == 2:
            zoom_factor = (ds_res / h, ds_res / w)
        elif len(image_data.shape) == 3:
            zoom_factor = (1, ds_res / h, ds_res / w)
        else:
            raise ValueError("Unexpected number of dimensions in image_data")

        image_data = zoom(image_data, zoom_factor, order=3)
        label_data = zoom(label_data, zoom_factor, order=0)
        out_path = os.path.join(output, ct_slice)
        np.savez(out_path, image=image_data, label=label_data)
    p_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data2/wyx/medical/amos22/datava/')
    parser.add_argument('--output', type=str, default='/data2/wyx/medical/amos22/datava/')
    parser.add_argument('--ds_res', type=int, default=224, help='Downsample resulution')
    args = parser.parse_args()
    main(args)