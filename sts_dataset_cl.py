import json
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from transforms import RandomResizedCropAndInterpolationWithTwoPic, _pil_interp
from timm.data import create_transform, ImageDataset 
from PIL import Image
import os
from transformers import BertTokenizer,AutoTokenizer
from beit_datasets import DataAugmentationForBEiT
from args import beit_args
from condenser import cl_data
class paired_dataset(Dataset):
    def __init__(self, json_path, tokenizer ,args):
        self.root = args.root
        with open(json_path,"r") as f:
            self.json_data=json.load(f)
        self.tokenizer = tokenizer
        self.transform = DataAugmentationForBEiT(args)
        self.args=args

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        _dict=self.json_data[index]
        img_path , caption =_dict["image"] , _dict["caption"]
        img_path = self.root + os.sep + img_path
        img = Image.open(img_path)
        beit_imput = self.transform(img)
        ids= self.tokenizer(
            caption,
            max_length=self.args.max_seq_length,
            truncation=True,
            padding=False,
        )
        return beit_imput, ids


if __name__=="__main__":
    tokenizer = BertTokenizer.from_pretrained("pretrained_model/condenser")
    txt_path = "/nlp_group/wuxing/suzhenpeng/beit2/ir_data/wiki1m.txt"
    # root = "/nlp_group/wuxing/ALBEF_downstream_dataset"
    # dataset = wiki1m_dataset(txt_path,tokenizer)
    # collator_fn = cl_data.CondenserCollator_text(dataset.tokenizer)
    # loader = DataLoader(dataset,collate_fn=collator_fn,batch_size=32)
    # k=0
    # for i in loader:
    #     k+=1
    #     print(k)

    bargs = beit_args.get_args()
    bargs.window_size=16
    json_path = "/nlp_group/wuxing/suzhenpeng/beit2_ratio_copy/ir_data/flickr_random_captions.json"
    dataset = paired_dataset(json_path, tokenizer, bargs)
    collator_fn = cl_data.imgDataCollatorWithPadding(tokenizer)
    loader = DataLoader(dataset,collate_fn=collator_fn,batch_size=4)
    for i in loader:
        print(i)
        print("a")