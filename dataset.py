import logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers_model.models.bert.tokenization_bert import BertTokenizer

class ShortTextDataset(Dataset):
    def __init__(self, tokenized_list, label_list):
        self.tokeized_list = tokenized_list
        self.label_list = label_list

    def __getitem__(self, item):
        return self.tokeized_list[item]["input_ids"],\
            self.tokeized_list[item]["attention_mask"],\
            self.tokeized_list[item]["token_type_ids"],\
            torch.tensor(int(self.label_list[item]))

    def __len__(self):
        return len(self.tokeized_list)


def get_short_text_dataset(config):
    train_set, eval_set, test_set = load_dataset(config)
    return train_set, eval_set, test_set


def load_dataset(config):
    CURRENTDIR = config["CURRENT_DIR"]
    # TRAIN_CACHED_PATH = os.path.join(CURRENTDIR, "data/train/train_cached.bin")
    # EVAL_CACHED_PATH = os.path.join(CURRENTDIR, "data/eval/eval_cached.bin")
    # TEST_CACHED_PATH = os.path.join(CURRENTDIR, "data/test/test_cached.bin")
    #
    # if os.path.exists(TEST_CACHED_PATH):
    #     torch.load()

    TRAIN_SOURCE_PATH = os.path.join(CURRENTDIR, config["TRAIN_DIR"])
    EVAL_SOURCE_PATH = os.path.join(CURRENTDIR, config["EVAL_DIR"])
    TEST_SOURCE_PATH = os.path.join(CURRENTDIR, config["TEST_DIR"])

    tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
    logger.info("构建数据集...")
    train_set = construct_dataset(tokenizer, TRAIN_SOURCE_PATH)
    eval_set = construct_dataset(tokenizer, EVAL_SOURCE_PATH)
    test_set = construct_dataset(tokenizer, TEST_SOURCE_PATH)
    logger.info("构建完成...训练集{}条数据，验证集{}条数据，测试集{}条数据...".format(len(train_set), len(eval_set), len(test_set)))
    return train_set, eval_set, test_set

def construct_dataset(tokenizer, SOURCE_PATH):
    tokenized_list = list()
    label_list = list()
    with open(SOURCE_PATH, "r", encoding="utf-8") as train_set:
        data = train_set.readlines()
        for line in tqdm(data):
            line = line.strip().split("\t")
            sequence = line[1]
            if line[2] == "是":
                label = 1
            else:
                label = 0
            
            # print(sequence, label)
            # break
            sequence_tokenized = tokenizer(sequence, return_tensors="pt", max_length=20, padding="max_length",
                                           truncation=True)
            tokenized_list.append(sequence_tokenized)
            label_list.append(label)

    return ShortTextDataset(tokenized_list, label_list)