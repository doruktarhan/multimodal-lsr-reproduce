- put the zip file of images under image_data folder
unzip the files with the folloiwng codes
cd image_data
unzip '*.zip'

- put the metadata file under meta_data folder

- the global variable image file path must be given with the unzipped image file path 'image_data/flickr30k-images' in the embedding_inference.py

- metadata files must be put in the correct directory or the directory must be changed in the main function of embedding_inference.py


- run the code with following
python embedding_inference.py --dataset_name mscoco --model_name clip --batch_size 128 --save_dir embeddings

- To run clip, change dense size to 512 in the D2SConfig in model.py.
