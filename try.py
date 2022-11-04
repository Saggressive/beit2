import random
import json
if __name__=="__main__":
    with open("ir_data/flickr_random_captions.json","r") as f:
        flickr=json.load(f)
    with open("ir_data/coco_random_captions.json","r") as f:
        coco=json.load(f)
    coco.extend(flickr)

    with open("ir_data/flickr_coco_random.json","w") as f:
        json.dump(coco, f)