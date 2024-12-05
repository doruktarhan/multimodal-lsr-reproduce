from loaders.dataset import MSCOCOdataset

meta_data_path = 'data/dataset_coco.json'

dataset = MSCOCOdataset(meta_data_path)
image_caption_pairs = dataset.get_image_caption_pairs()

