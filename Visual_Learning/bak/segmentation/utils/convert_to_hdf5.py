import sys
import os
from random import sample
from tqdm import tqdm
import numpy as np
import json
import argparse
import h5py
from glob import glob

def main(args):
    split_file_path = os.path.join('dataset', args.split_file)
    if not os.path.exists(split_file_path):
        raise Exception('dataset path % s does not exist' % split_file_path)
    catalog = []
    f_ = open(split_file_path,"r")
    n_lines = 0
    for line in f_:
        n_lines += 1
    f_.close()
    file_counter = 0
    f_ = open(split_file_path,"r")
    for line in f_:
        if file_counter < n_lines/5:
            name, num = line.split(" ")
            num = num.rstrip()
            image_path = "dataset/lfw_funneled/"+name+"/"+name+"_"+num.zfill(4)+".jpg"
            gt_path = "dataset/parts_lfw_funneled_gt_images/"+name+"_"+num.zfill(4)+".ppm"
            if not os.path.isfile(gt_path) or not os.path.isfile(image_path):
                print (image_path, "does not exist")
            else:
                catalog.append({'image_path': image_path, 'label_path':gt_path})
        file_counter += 1
    f_.close()
    # split training and evaluation
    binary_dir = os.path.join('dataset')
    out_f = args.split_file.split('_')
    out_f = out_f[1][:-4] + '.hdf5'
    if not os.path.exists(binary_dir):
        os.makedirs(binary_dir)
    generate_binary_file(catalog, os.path.join(binary_dir, out_f))

def generate_binary_file(dataset_catalog, save_path):
    hf = h5py.File(save_path, 'a')
    n_samples = len(dataset_catalog)
    print(n_samples)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    image_data = hf.create_dataset('images', (n_samples, ), dtype=dt)
    label_data = hf.create_dataset('labels', (n_samples, ), dtype=dt)
    for idx, data_pair in tqdm(enumerate(dataset_catalog)):
        img_f = open(data_pair["image_path"], 'rb')
        img_binary = img_f.read()
        image_data[idx] = np.frombuffer(img_binary, dtype='uint8')
        label_f = open(data_pair["label_path"], 'rb')
        label_binary = label_f.read()
        label_data[idx] = np.frombuffer(label_binary, dtype='uint8')
    hf.close()

if __name__ == '__main__':
    generator_parser = argparse.ArgumentParser(
        description='Split dataset for trainig and evaluation')
    generator_parser.add_argument('-f', '--split_file', type=str, default='')
    args = generator_parser.parse_args()
    main(args)

