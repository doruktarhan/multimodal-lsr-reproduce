import os
import json
from tqdm import tqdm
from PIL import Image


# Global variable for image folder
PATH_TO_MSCOCO_FOLDER = '/Users/doruktarhan/Desktop/MSCOCO_Dataset'
PATH_TO_FLICKR_FOLDER = '/Users/doruktarhan/Desktop/Flickr30k/flickr30k-images'

class MSCOCOdataset:
    def __init__(self,meta_data_path: str):
        self.meta_data_path = meta_data_path
        self.meta_data = None
        self.image_caption_pairs = []
        self.load_meta_data() # Load meta data at initialization

    def load_meta_data(self):
        try:
            with open(self.meta_data_path, 'r') as file:
                self.meta_data = json.load(file)
        except FileNotFoundError:
            print(f"File not found at {self.meta_data_path}")
        except Exception as e:
            print(f"An error occured while loading meta data: {e}")

    def get_image_caption_pairs(self):
        if not self.meta_data:
            raise ValueError("Meta data is not loaded. Error on initlization of the dataset.")
        
        for image_data in tqdm(self.meta_data['images'], desc="Processing images and captions"):
            #save the image id
            image_id = image_data['imgid']
            #save the image path
            image_folder = image_data['filepath']
            image_name = image_data['filename']
            image_path = os.path.join(PATH_TO_MSCOCO_FOLDER, image_folder)
            image_path = os.path.join(image_path, image_name)

            #get the image
            try:
                image = Image.open(image_path)
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
                continue
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            for sentence_meta_data in image_data['sentences']:
                #get the sentence_id
                sentence_id = sentence_meta_data['sentid']
                #get the sentence
                caption = sentence_meta_data['raw']
                self.image_caption_pairs.append((image_id, sentence_id, image_path, caption))

        return self.image_caption_pairs


        
class Flickr30kdataset:
    def __init__(self,meta_data_path: str):
        self.meta_data_path = meta_data_path
        self.meta_data = None
        self.image_caption_pairs = []
        self.load_meta_data() # Load meta data at initialization

    def load_meta_data(self):
        try:
            with open(self.meta_data_path, 'r') as file:
                self.meta_data = json.load(file)
        except FileNotFoundError:
            print(f"File not found at {self.meta_data_path}")
        except Exception as e:
            print(f"An error occured while loading meta data: {e}")

    def get_image_caption_pairs(self):
        if not self.meta_data:
            raise ValueError("Meta data is not loaded. Error on initlization of the dataset.")
        

        for image_data in tqdm(self.meta_data['images'], desc="Processing images and captions"):
            #save the image id
            image_id = image_data['imgid']
            #save the image path
            image_name = image_data['filename']
            image_path = os.path.join(PATH_TO_FLICKR_FOLDER,image_name)

            #get the image
            try:
                image = Image.open(image_path)
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
                continue
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            for sentence_meta_data in image_data['sentences']:
                #get the sentence_id
                sentence_id = sentence_meta_data['sentid']
                #get the sentence
                caption = sentence_meta_data['raw']
                self.image_caption_pairs.append((image_id, sentence_id, image_path, caption))
        print(f"Number of image-caption pairs: {len(self.image_caption_pairs)}")
        print(f"Type of image-caption pairs: {type(self.image_caption_pairs)}")
        print(f"First element of image-caption pairs: {self.image_caption_pairs[0]}")
        return self.image_caption_pairs
