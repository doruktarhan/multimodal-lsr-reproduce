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

- For the LSR task, you need to run the train file. Select the model and dataset from huggingface datasets repo. 
https://huggingface.co/doruktarhan6

- To train LSR use the folloiwn command and adjust the parameters. 

python train.py --data doruktarhan6/flickr30k-clip-dense --train_batch_size 512 --eval_batch_size 1024  --q_reg 0.001 --d_reg 0.001  --temp 0.001 --use_amp --epochs 200 