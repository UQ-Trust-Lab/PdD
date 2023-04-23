from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
from new_version.perturbation import *
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils import train, test, MyDataSet
from sklearn import metrics

# Meta data
NUM_LABELS = 2
MODEL_NAME = "bert-base-uncased"
DATA_SET = "rotten_tomatoes"
DISTRIBUTION = "uniform"
DENSITY = 0.5
DIVERSITY_DICT = {' ': [' ']}
DIVERSITY_DICT.update(DELETION_DICT)
EPOCH = 10
# Initialise model and tokenizer from meta data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
# Load the data
dataset = load_dataset(DATA_SET)
train_set = dataset["train"]
val_set = dataset["validation"]
test_set = dataset["test"]
# Initialise the perturbation generator
generator = Generator(DISTRIBUTION, DENSITY, DIVERSITY_DICT)
train_set_text = train_set["text"]
train_set_label = train_set["label"]
test_set_text = val_set["text"] + test_set["text"]
test_set_label = val_set["label"] + test_set["label"]
# Add perturbation to the data
perturbed_train_set_text = []
for i in range(len(train_set_text)):
    perturbed_train_set_text.append(generator.generate(train_set_text[i]))
