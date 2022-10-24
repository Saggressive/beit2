# Copyright 2021 Condenser Author All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask
from copy import deepcopy
from beit_datasets import DataAugmentationForBEiT

@dataclass
class CondenserCollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512

    def __init__(self,beit_args,max_seq_length,**kwargs):
        super(CondenserCollator, self).__init__(**kwargs)
        self.beit_args=beit_args
        self.max_seq_length=max_seq_length
        self.transform= DataAugmentationForBEiT(beit_args)

    def __post_init__(self):
        super(CondenserCollator, self).__post_init__()

        from transformers import BertTokenizer, BertTokenizerFast
        from transformers import RobertaTokenizer, RobertaTokenizerFast
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_bert
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            self.whole_word_cand_indexes = self. _whole_word_cand_indexes_roberta
        else:
            raise NotImplementedError(f'{type(self.tokenizer)} collator not supported yet')

        self.specials = self.tokenizer.all_special_tokens

    def _whole_word_cand_indexes_bert(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_cand_indexes_roberta(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                raise ValueError('We expect only raw input for roberta for current implementation')

            if i == 0:
                cand_indexes.append([0])
            elif not token.startswith('\u0120'):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = self._whole_word_cand_indexes_bert(input_tokens)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _truncate(self, example: List[int]):
        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = random.randint(0, trunc)
        trunc_right = trunc - trunc_left

        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]

        if not len(truncated) == tgt_len:
            print(len(example), len(truncated), trunc_left, trunc_right, tgt_len, flush=True)
            raise ValueError
        return truncated

    # def _truncate(self, example: List[int]):
    #     tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)
    #     if len(example) <= tgt_len:
    #         return example
    #     else:
    #         return example[:tgt_len]

    def _pad(self, seq, val=0):
        tgt_len = self.max_seq_length
        assert len(seq) <= tgt_len
        return seq + [val for _ in range(tgt_len - len(seq))]

    def __call__(self, examples):
        if self.beit_args.use_text_cl:
            new_examples=[]
            for img,caption in examples:
                new_examples.extend([((img[0],img[1],img[2]),caption),
                                ((img[0].clone(),img[1].clone(),img[2].copy()),deepcopy(caption))])
        examples=new_examples
        encoded_examples = []
        masks = []
        mlm_masks = []
        samples, images, bool_masked_pos = [] , [] , []
        token_type_ids = []
        for img,caption in examples:
            # img=self.transform(img)
            e_trunc = self._truncate(caption)
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
            mlm_mask = self._whole_word_mask(tokens)
            mlm_mask = self._pad([0] + mlm_mask)
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(
                text=self._truncate(caption),
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,)
            token_type_ids.append(encoded["token_type_ids"])
            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])
            samples.append(img[0])
            images.append(img[1])
            bool_masked_pos.append(img[2])

        mlm_inputs, mlm_labels = self.mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )

        batch = {
            "samples":torch.stack(samples),
            "images":torch.stack(images),
            "bool_masked_pos":torch.tensor(bool_masked_pos),
            "input_ids": torch.tensor(encoded_examples, dtype=torch.long),
            "mlm_input_ids":mlm_inputs,
            "mlm_labels": mlm_labels,
            "attention_mask": torch.tensor(masks),
            "token_type_ids":torch.tensor(token_type_ids,dtype=torch.long)
        }

        return batch


@dataclass
class CondenserCollator_text(CondenserCollator):
    
    def __init__(self,beit_args,max_seq_length,**kwargs):
        super(CondenserCollator, self).__init__(**kwargs)
        self.beit_args=beit_args
        self.max_seq_length=max_seq_length
        self.transform= DataAugmentationForBEiT(beit_args)

    def __call__(self, examples):
            
        if self.beit_args.use_text_cl:
            new_examples = []
            for i in range(len(examples)):
                caption = examples[i]
                while caption==[] or len(caption)==0 or caption is None:
                    other = random.randint(0,len(examples)-1)
                    caption = deepcopy(examples[other])
                new_examples.extend([caption,deepcopy(caption)])
        examples=new_examples
        encoded_examples = []
        masks = []
        mlm_masks = []
        token_type_ids = []
        for index,caption in enumerate(examples):
            
            e_trunc = self._truncate(caption)
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
            mlm_mask = self._whole_word_mask(tokens)
            mlm_mask = self._pad([0] + mlm_mask)
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(
                self._truncate(caption),
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,)

            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])
            token_type_ids.append(encoded["token_type_ids"])

        mlm_inputs, mlm_labels = self.mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )

        batch = {
            "input_ids": torch.tensor(encoded_examples, dtype=torch.long),
            "mlm_input_ids": mlm_inputs,
            "mlm_labels": mlm_labels,
            "attention_mask": torch.tensor(masks),
            "token_type_ids":torch.tensor(token_type_ids,dtype=torch.long)
        }

        return batch


class CoCondenserDataset(Dataset):
    def __init__(self, dataset, data_args):
        self.dataset = dataset
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        spans = self.dataset[item]['spans']
        return random.sample(spans, 2)