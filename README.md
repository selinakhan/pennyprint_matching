

# Automated Pennyprints Visual Similarity Matching

## Code files

This repository includes the code needed to process and compare annotated penny prints. It includes two files:

`process.py` - This file is used to process the annotated images and for each individual pane per image, generate it's _n_ nearest neighbours. 

`results.py` - This file is used to generate a visual representation of the compared image panes.

`requirements.txt` - This file includes the required packages to properly run the code.

## Data and Directories 

The code assumes some pre-defined files to work properly. Below is a suggested directory set-up which will help to structure the generated results. 

`images` - This directory all the images of the pennyprints. The filenames must correspond to the filenames specified in the bounding box annotations file. 

`annotations` - This directory includes the `.json` file with the bounding box annotations for the penny print images.

`processed_images` - This directory is initially empty and will be used to save the individual panes per penny print.

`processed_data` - This directory is initially empty and will be used to store the extracted data needed for the pennyprint comparison. 

`results` - This directory will also be initially empty and is used to store the results.

## Code Usage

**To process and compare the pennyprint images, the following command is used.**

`python3 process.py --im_in_dir [path to original images] --im_out_dir [path to store processed images] --data_in_dir [path to the annotations file] --data_out_dir [path to store the extracted data] --neighbours [integer] --augmentation [augmentation type]`

The `--neighbours` argument specifies how many similar images should be calculated per image pane. This is sorted: meaning that _n=5_ will generate the top-5 similar images.

The `--augmentation` argument specifies which data augmentation is used on the panes before they are processed. Different image augmentations bring out different features in an image. Choose from: 
- gray: displays the image in grayscale. It simplifies the comparison between images by neglecting hue changes. 
- edge: displays the 'edges' in an image. It highlights the transitions between different colors in an image. 
- binary: displays the image in strictly black and white colors. It highlights the complementing hues and shades in an image.
- None: No image augmentation is applied. 

Visual examples are dispayed below. ![here](https://github.com/selinakhan/pennyprint_matching/blob/main/augmentations.png) 
From top to bottom: gray, edge, binary.

Default is *binary*, which in experiments showed best results. 

### Data files stored

The `--data_out_dir` directory stores the following files:

 - `mapping.json` - This file serves as a mapping from image/pane ID to filename of the original image.
 - `similars.json` - This file stores the similar image panes per image pane.
 - `processed_data.json` - This file stored the original bounding box annotations per image including an assigned image ID.
 - `features.pkl` - This file can be ignored. It stores the image features.

All saved processed images are annotated the following way:

`img1_p4.jpeg`  -  indicating the image ID and pane number. Individual panes are annotated for each image counting from left to right, top to down.  

**To generate (visual representations of the) results for a single image or all images, the following command is used.**

`python3 results.py --proc_im_in_dir [path to processed images] --mapping [path to mapping.json] --similars [path to similars.json] --data_out_dir [path to store results] --all_imgs <flag>`

if the `--all_imgs` argument is not flagged, the commandline will promt the user to specify an individual image and which panes in the image should generate the results. If a single pane should be generated, `from pane`  and `to pane` is equal. 
If `--all_imgs` is flagged, the results for all images and all panes are generated.

### Results stored

The results are stored in the specified `--data_out_dir`. For each image a seperate directory is made, for which each pane per image has its neighbours stored in a seperate directory. Example of how the storage works is shown below:

```
results
└─── penny_print_image
│   └───p0
│       │   match_1.jpeg
│       │   match_2.jpeg
│       │   original_p0.jpeg
│       │   ...
│       │   results_p0.jpeg
│   └───p1
│       │   match_1.jpeg
│       │   match_2.jpeg
│       │   original_p1.jpeg
│       │   ...
│       │   results_p1.jpeg
│   ...
```

For a single pennyprint image, the results of the first two panes are generated. In each pane sub-folder the first _n_ similar images are stored, along with the `.txt` file mapping each similar image to its original image and pane number.

 
