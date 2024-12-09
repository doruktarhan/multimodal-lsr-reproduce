- put the zip file of images under image_data folder
unzip the files with the folloiwng codes
cd image_data
unzip '*.zip'

- put the metadata file under meta_data folder

- the global variable image file path must be given with the unzipped image file path
 'image_data/flickr30k-images'

- run the code with following
python embedding_inference.py --dataset_name flickr30k --model_name blip --batch_size 16 --save_dir embeddings
