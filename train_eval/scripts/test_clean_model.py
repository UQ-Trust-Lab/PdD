"""Test the clean model's performance on perturbed datasets"""

import pickle
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import test, MyDataSet, set_parameters
from sklearn import metrics
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser(description='PDD TESTER')
parser.add_argument('--perturbation', default="deletion", type=str,
                    help='perturbation method')
parser.add_argument('--distribution', default="uniform", type=str,
                    help='distribution of perturbation')
parser.add_argument('--density', default=0.05, type=float,
                    help='density of perturbation')
parser.add_argument('--model', default="bert-base-uncased", type=str,
                    help='large language model')
parser.add_argument('--dataset', default="rotten_tomatoes", type=str,
                    help='dataset for task')
args = parser.parse_args()

# Meta data
NUM_LABELS = 2
MODEL_NAME = args.model
DATA_SET = args.dataset
PERTURBATION = args.perturbation
DISTRIBUTION = args.distribution
DENSITY = args.density
NUM_OF_PERTURBATION = 10
RESULT_FILE = f"../results/{DATA_SET}/{PERTURBATION}/{MODEL_NAME}_clean_testing.txt"
CLEAN_MODEL = f"../models/{DATA_SET}/{MODEL_NAME}_clean_training"
DATASET_FILE = f"../perturbed_datasets/{DATA_SET}/{PERTURBATION}/{PERTURBATION}_{DISTRIBUTION}_{DENSITY}_test.csv"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
clean_model_params = pickle.load(open(CLEAN_MODEL, "rb"))
model = set_parameters(model, clean_model_params)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

with open(RESULT_FILE, "a") as file:
    file.write(f"Model: {MODEL_NAME} clean\n")
    file.write(f"Dataset: {DATA_SET}\n")
    file.write(f"Perturbation: {PERTURBATION}\n")
    file.write(f"Distribution: {DISTRIBUTION}\n")
    file.write(f"Density: {DENSITY}\n")
    file.write(f"Number of perturbation: {NUM_OF_PERTURBATION}\n")

# Load the perturbed dataset
perturbed_test_set = pd.read_csv(DATASET_FILE)
perturbed_test_text = perturbed_test_set["text"].to_list()
perturbed_test_label = perturbed_test_set["label"].to_list()

# Tokenize the data and create tensors
tokenized_test_set = tokenizer(perturbed_test_text, truncation=True, padding=True, return_tensors="pt")
tokenized_test_set["labels"] = torch.LongTensor(perturbed_test_label).clone()
tokenized_test_set = MyDataSet(tokenized_test_set)

# Create data loaders
test_loader = DataLoader(tokenized_test_set, shuffle=True, batch_size=128)

# Test the model
test_loop = tqdm(test_loader, leave=True)
overall_test_loss = 0
epoch_test_predictions = None
epoch_test_labels = None
num_of_test_batches = len(test_loader)
model.to(device)
for test_batch in test_loop:
    model.eval()
    test_loss, test_predictions, test_accuracy, test_labels = test(test_batch, model, device)
    test_loop.set_postfix(test_loss=test_loss.item(), test_accuracy=test_accuracy.item())
    overall_test_loss += test_loss.item()
    if epoch_test_predictions is None:
        epoch_test_predictions = test_predictions
        epoch_test_labels = test_labels
    else:
        epoch_test_predictions = torch.cat((epoch_test_predictions, test_predictions), dim=0)
        epoch_test_labels = torch.cat((epoch_test_labels, test_labels), dim=0)

average_test_loss = overall_test_loss / num_of_test_batches
epoch_test_accuracy = torch.sum(torch.eq(epoch_test_predictions, epoch_test_labels)) / epoch_test_labels.shape[0]

average_test_precision = metrics.precision_score(torch.flatten(epoch_test_labels).tolist(),
                                                 torch.flatten(epoch_test_predictions).tolist(), average="macro")
average_test_recall = metrics.recall_score(torch.flatten(epoch_test_labels).tolist(),
                                           torch.flatten(epoch_test_predictions).tolist(), average="macro")
average_test_f1 = metrics.f1_score(torch.flatten(epoch_test_labels).tolist(),
                                   torch.flatten(epoch_test_predictions).tolist(), average="macro")

print(
    f"test loss: {average_test_loss} accuracy: {epoch_test_accuracy} precision: {average_test_precision} recall: {average_test_recall} f1: {average_test_f1}")

print(metrics.classification_report(torch.flatten(epoch_test_labels).tolist(),
                                    torch.flatten(epoch_test_predictions).tolist()))

with open(RESULT_FILE, "a") as file:
    file.write(
        f"test loss: {average_test_loss} accuracy: {epoch_test_accuracy} precision: {average_test_precision} recall: {average_test_recall} f1: {average_test_f1}\n")
    file.write("-----------------------------------------------------\n")
    file.write("\n")
