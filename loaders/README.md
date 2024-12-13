- put the zip file of images under image_data folder
unzip the files with the folloiwng codes
cd image_data
unzip '*.zip'

- put the metadata file under meta_data folder

- the global variable image file path must be given with the unzipped image file path
 'image_data/flickr30k-images'

- run the code with following
python embedding_inference.py --dataset_name mscoco --model_name blip2avg --batch_size 128 --save_dir embeddings
