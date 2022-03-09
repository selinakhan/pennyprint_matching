import os
import json
import tqdm
import torch
import torchvision.transforms as T

import clip
import cv2 as cv
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import argparse

from ast import excepthandler
from typing import Mapping
from tqdm import trange, tqdm
from PIL import Image, ImageOps, ImageFile
from PIL import UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics.pairwise import cosine_similarity


def process_data(in_dir, out_dir, out_data, data):
    ''' Load images, extract panes and save as seperate images with
        corresponding ID's. Save processed data including mapping from ID to image(panes)'''

    data = json.load(open(data))
    mapping = {}

    for im_id, image in enumerate(tqdm(data['_via_img_metadata'])):
        filename = data['_via_img_metadata'][image]['filename']

        try:
            im = img.imread(in_dir + filename)
            data['_via_img_metadata'][image]['id'] = im_id
        except UnidentifiedImageError:
            print(f'Image: {image} is corrupted')
            continue

        for box_id, bb in enumerate(data['_via_img_metadata'][image]['regions']):
            box = bb['shape_attributes']
            box['box_id'] = box_id

            bbox = im[box['y']:box['y'] + box['height'], box['x']:box['x'] + box['width'], :]

            mapping[f'img{im_id}_p{box_id}'] = filename

            plt.imsave(out_dir + f'img{im_id}_p{box_id}.jpeg', bbox)

    with open(out_data + 'processed_data.json', 'w') as f:
        json.dump(data, f)

    with open(out_data + 'mapping.json', 'w') as f:
        json.dump(mapping, f)
        
    

def extract_features(in_dir, out_dir, model, preprocess, device, augmentation):
    ''' Load panes and extract CLIP features. Return .json file
        with feature vectors per pane.'''

    all_features = {}

    for image in tqdm(os.listdir(in_dir)):
        key = image.split('.')[0]

        im = Image.open(in_dir + image)

        if augmentation == 'gray':
            transform = T.Compose([T.Grayscale(3)])
            im = transform(im)

        if augmentation == 'edge':
            im = cv.Canny(np.array(im), 250, 350)
            im = Image.fromarray(im)

        if augmentation == 'binary':
            im = cv.threshold(np.array(ImageOps.grayscale(im)), 128, 255, cv.THRESH_BINARY)[1]
            im = Image.fromarray(im)

        img_pr = preprocess(im).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img_pr)

        all_features[key] = image_features.cpu()

    with open(f"{out_dir}features.pkl", "wb+") as f:
        pickle.dump(all_features, f)


def load_features(img_features):
    with open(img_features, "rb") as f:
        all_features = pickle.load(f)
    
    # Center mean
    data = torch.stack(list(all_features.values())).squeeze(1)
    mean = torch.mean(data, dim=0)

    for key in all_features:
        all_features[key] -= mean

    return all_features, data


def process_similars(im_dir, img_features, out_file, neigbours):
    ''' Use pane features and return .json file with each 
        pane and its similars. '''

    data, fts = load_features(img_features)

    similar_imgs = {}

    mapping = np.array(list(data.keys()))

    matrix = cosine_similarity(fts)

    np.fill_diagonal(matrix, -1)

    for i, row in enumerate(tqdm(matrix)):
        im = mapping[i].split('_')[0]
        output = np.array([idx for idx, element in enumerate(mapping) if mapping[idx].split('_')[0] == im])
        row[output] = -1
        similars = np.argsort(row)[-neigbours:]
        similar_imgs[mapping[i]] = list(np.take(mapping, similars))
        

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(similar_imgs, f, ensure_ascii=False, indent=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--im_in_dir', default='data/images/', type=str, required=False, help="Path to original images")
    parser.add_argument('--im_out_dir', default='data/processed_images/', type=str, required=False, help='Path to store image patches')
    parser.add_argument('--data_in_dir', default='data/labels/[Updated] Bounding boxes 14-12.json', type=str, required=False, help='Path to bounding box data')
    parser.add_argument('--data_out_dir', default='processed_data/', type=str, required=False, help='Path to store processes data (include mappings)')

    parser.add_argument('--neighbours', default=5, type=int, required=False, help='Show n amount of nearest neighbours')
    parser.add_argument('--augmentation', default=None, type=str, required=False, help='Data augmentation to apply')

    args = parser.parse_args()

    if args.augmentation not in ['gray', 'edge', 'binary', None]:
        raise ValueError(f'Invalid augmentation specification: {args.augmentation}. Select from: gray, edge, binary')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Loading model...')
    print()
    model, preprocess = clip.load("ViT-B/32", device=device)

    print(f'Running on: {device}') # Use GPU if available, else use CPU
    print()


    print('Processing data...')
    print()
    process_data(args.im_in_dir, args.im_out_dir, args.data_out_dir, args.data_in_dir)

    print('Extracting image features...')
    print()
    extract_features(args.im_out_dir, args.data_out_dir, model, preprocess, device, args.augmentation)

    print('Processing similar panes...')
    print()
    process_similars(args.im_out_dir, f'{args.data_out_dir}/features.pkl', f'{args.data_out_dir}similars.json', args.neighbours)

    print('Finished!')
