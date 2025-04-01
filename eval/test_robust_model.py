import pickle
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast

from eval.utils import *

# Metadata
parser = ArgumentParser(description="PDD TESTER")
parser.add_argument(
    "--perturbation", default="deletion", type=str, help="perturbation method"
)
parser.add_argument(
    "--distribution", default="uniform", type=str, help="distribution of perturbation"
)
parser.add_argument(
    "--density", default=0.05, type=float, help="density of perturbation"
)
parser.add_argument(
    "--model", default="bert-base-uncased", type=str, help="large language model"
)
parser.add_argument(
    "--dataset", default="rotten_tomatoes", type=str, help="dataset for task"
)
parser.add_argument("--device", default="cuda:0", type=str, help="cuda device")

args = parser.parse_args()

NUM_LABELS = 2
MODEL_NAME = args.model
DATA_SET = args.dataset
PERTURBATION = args.perturbation
DISTRIBUTION = args.distribution
DENSITY = args.density
NUM_OF_PERTURBATION = 10
DEVICE = args.device
RESULT_FILE = f"../results/{DATA_SET}/{PERTURBATION}/{MODEL_NAME}_robust_testing.txt"
MODEL_PATH = (
    f"../models/{DATA_SET}/{PERTURBATION}/"
    f"{MODEL_NAME}_{PERTURBATION}_{DISTRIBUTION}_{DENSITY}"
)

# Initialise model and tokenizer from metadata
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model_params = pickle.load(open(MODEL_PATH, "rb"))
model = set_parameters(model, model_params)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

with open(RESULT_FILE, "a") as file:
    file.write(f"Model: {MODEL_NAME}\n")
    file.write(f"Dataset: {DATA_SET} clean\n")
    file.write(f"Perturbation: {PERTURBATION}\n")
    file.write(f"Distribution: {DISTRIBUTION}\n")
    file.write(f"Density: {DENSITY}\n")
    file.write(f"Number of perturbation: {NUM_OF_PERTURBATION}\n")

# Load the data
dataset = load_dataset(DATA_SET)
val_set = dataset["validation"]
test_set = dataset["test"]
# Tokenize the data and create tensors
# Combine val and test set and one test set
tokenized_test_set = tokenizer(
    val_set["text"] + test_set["text"],
    truncation=True,
    padding=True,
    return_tensors="pt",
)
tokenized_test_set["labels"] = torch.LongTensor(
    val_set["label"] + test_set["label"]
).clone()
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
    test_loss, test_predictions, test_accuracy, test_labels = test(
        test_batch, model, device
    )
    test_loop.set_postfix(
        test_loss=test_loss.item(), test_accuracy=test_accuracy.item()
    )
    overall_test_loss += test_loss.item()
    if epoch_test_predictions is None:
        epoch_test_predictions = test_predictions
        epoch_test_labels = test_labels
    else:
        epoch_test_predictions = torch.cat(
            (epoch_test_predictions, test_predictions), dim=0
        )
        epoch_test_labels = torch.cat((epoch_test_labels, test_labels), dim=0)

average_test_loss = overall_test_loss / num_of_test_batches
epoch_test_accuracy = (
    torch.sum(torch.eq(epoch_test_predictions, epoch_test_labels))
    / epoch_test_labels.shape[0]
)

average_test_precision = metrics.precision_score(
    torch.flatten(epoch_test_labels).tolist(),
    torch.flatten(epoch_test_predictions).tolist(),
    average="macro",
)
average_test_recall = metrics.recall_score(
    torch.flatten(epoch_test_labels).tolist(),
    torch.flatten(epoch_test_predictions).tolist(),
    average="macro",
)
average_test_f1 = metrics.f1_score(
    torch.flatten(epoch_test_labels).tolist(),
    torch.flatten(epoch_test_predictions).tolist(),
    average="macro",
)

print(
    f"test loss: {average_test_loss} "
    f"accuracy: {epoch_test_accuracy} "
    f"precision: {average_test_precision} "
    f"recall: {average_test_recall} "
    f"f1: {average_test_f1}"
)

print(
    metrics.classification_report(
        torch.flatten(epoch_test_labels).tolist(),
        torch.flatten(epoch_test_predictions).tolist(),
    )
)

with open(RESULT_FILE, "a") as file:
    file.write(
        f"test loss: {average_test_loss} "
        f"accuracy: {epoch_test_accuracy} "
        f"precision: {average_test_precision} "
        f"recall: {average_test_recall} "
        f"f1: {average_test_f1}\n"
    )
    file.write("-----------------------------------------------------\n")
    file.write("\n")
