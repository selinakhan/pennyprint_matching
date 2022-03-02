import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import argparse
from PIL import Image

def visualise_results(im_dir, proc_im_dir, out_dir, mapping, similars, image, pane_from, pane_to):
    im = image.split('.jpeg')[0]

    if not os.path.exists(f'{out_dir}/{im}'):
        os.mkdir(f'{out_dir}/{im}')

    im_dict = {k:v for k, v in mapping.items() if v == image}
    keys = list(im_dict)[pane_from - 1:pane_to]

    for key in keys:
        res = {}
        original = Image.open(f'{proc_im_dir}{key}.jpeg')

        save_k = key.split('_')[1]

        if not os.path.exists(f'{out_dir}/{im}/{save_k}'):
            os.mkdir(f'{out_dir}/{im}/{save_k}')

        matches = []
        for match in similars[key]:
            matches.append(Image.open(f'{proc_im_dir}{match}.jpeg'))

        matches = list(reversed(matches))

        res[save_k] = [(str(mapping[img]), img.split('_')[1]) for img in similars[key]]

        # Save original image
        original.save(f'{out_dir}/{im}/{save_k}/original_{save_k}.jpeg')

        for i, match in enumerate(matches):
            idx = i + 1
            matches[-i].save(f'{out_dir}/{im}/{save_k}/match_{idx}.jpeg')

        with open(f'{out_dir}/{im}/{save_k}/results_{save_k}', 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=3)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--im_in_dir', default='data/images/', type=str, required=False, help="Path to original images")
    parser.add_argument('--proc_im_dir', default='data/processed_images/', type=str, required=False, help="Path to processed panes")

    parser.add_argument('--mapping', default='processed_data/mapping.json', type=str, required=False, help='Path to image-pane ID mappings')
    parser.add_argument('--similars', default='processed_data/similars.json', type=str, required=False, help='Path to similarity results')
    parser.add_argument('--data_out_dir', default='results/', type=str, required=False, help='Path to store results')
    parser.add_argument('--all_imgs', action='store_true', required=False, help='Process all images at once')

    args = parser.parse_args()


    similars = json.load(open(args.similars))
    mapping = json.load(open(args.mapping))


    if not args.all_imgs:
        image = input("Filename: ")
        pane_from = int(input("From pane: "))
        pane_to = int(input("To pane: "))
        
        # 1024px-Aanschouwt_hoe_ieder_hier_zijn_waren_weet_te_pryzen-Catchpenny_print-Borms_0834.jpeg
        # 3

        visualise_results(args.im_in_dir, args.proc_im_dir, args.data_out_dir, mapping, similars, image, pane_from, pane_to)

    else:
        print('Processing all available images...')
        unique_imgs = set(val for val in mapping.values())
        res = dict(Counter(mapping.values()))

        for img in tqdm(unique_imgs):
            visualise_results(args.im_in_dir, args.proc_im_dir, args.data_out_dir, mapping, similars, img, 1, res[img])




