import csv
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict

import torch
from datasets import load_dataset
from sklearn import metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from eval.utils import *
from pdd import *

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
    "--model", default="roberta-base", type=str, help="large language model"
)
parser.add_argument("--dataset", default="snli", type=str, help="dataset for task")
parser.add_argument("--device", default="cuda:0", type=str, help="cuda device")

args = parser.parse_args()

random.seed(42)
# Meta data
NUM_LABELS = 4
MODEL_NAME = args.model
DATA_SET = args.dataset
PERTURBATION = args.perturbation
DISTRIBUTION = args.distribution
DENSITY = args.density
DEVICE = args.device
NUM_OF_PERTURBATION = 10

RESULT_FILE = (
    f"../results/{DATA_SET}/{PERTURBATION}/"
    f"{MODEL_NAME}_{PERTURBATION}_{DISTRIBUTION}_{DENSITY}_robust_training.txt"
)
CLEAN_MODEL = f"../models/{DATA_SET}/{MODEL_NAME}_clean_training"
with open(RESULT_FILE, "a") as file:
    file.write(f"Model: {MODEL_NAME}\n")
    file.write(f"Dataset: {DATA_SET}\n")
    file.write(f"Perturbation: {PERTURBATION}\n")
    file.write(f"Distribution: {DISTRIBUTION}\n")
    file.write(f"Density: {DENSITY}\n")
    file.write(f"Number of perturbation: {NUM_OF_PERTURBATION}\n")

diversity_dict = defaultdict(lambda: [" "])

PERTURBATION_DICT = {
    "deletion": DELETION_DICT,
    "typo": KEYBOARD_TYPO_ADVANCED_DICT,
    "homoglyphs": HOMOGLYPHS_DICT,
    "invisible": INVISIBLE_UNICODE_DICT,
}

diversity_dict.update(PERTURBATION_DICT[PERTURBATION])

# Initialise model and tokenizer from meta data
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)
clean_model_params = pickle.load(open(CLEAN_MODEL, "rb"))
model = set_parameters(model, clean_model_params)
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

# Load the data
dataset = load_dataset(DATA_SET)
train_set = dataset["train"]
val_set = dataset["validation"]
test_set = dataset["test"]

train_premises = train_set["premise"]
train_hypotheses = train_set["hypothesis"]
train_text = []
for i in range(len(train_premises)):
    text = [train_premises[i], train_hypotheses[i]]
    train_text.append(text)
train_labels = train_set["label"]
# This step is to shift the -1 label to 0 so that no error would occur
train_labels = [label + 1 for label in train_labels]

val_premises = val_set["premise"]
val_hypotheses = val_set["hypothesis"]
val_text = []
for i in range(len(val_premises)):
    text = [val_premises[i], val_hypotheses[i]]
    val_text.append(text)
val_labels = val_set["label"]
val_labels = [label + 1 for label in val_labels]

test_premises = test_set["premise"]
test_hypotheses = test_set["hypothesis"]
test_text = []
for i in range(len(test_premises)):
    text = [test_premises[i], test_hypotheses[i]]
    test_text.append(text)
test_labels = test_set["label"]
test_labels = [label + 1 for label in test_labels]

test_text = val_text + test_text
test_labels = val_labels + test_labels

# Randomly sample 9k data from the training set and 2k data from test set to match the
# dataset size of rotten tomatoes
tran_indexes = random.sample(range(len(train_text)), 2)
train_text = [train_text[i] for i in tran_indexes]
train_labels = [train_labels[i] for i in tran_indexes]

test_indexes = random.sample(range(len(test_text)), 2)
test_text = [test_text[i] for i in test_indexes]
test_labels = [test_labels[i] for i in test_indexes]

# Add perturbation to the data
perturbed_train_set_text = []
for i in range(len(train_text)):
    for k in range(NUM_OF_PERTURBATION):
        perturbed_text = []
        for j in range(len(train_text[i])):
            generator = Generator(DISTRIBUTION, DENSITY, diversity_dict)
            perturbed_text.append(generator.generate(train_text[i][j]))
        perturbed_train_set_text.append(perturbed_text)

perturbed_test_set_text = []
for i in range(len(test_set_text)):
    for j in range(NUM_OF_PERTURBATION):
        generator = Generator(DISTRIBUTION, DENSITY, diversity_dict)
        perturbed_test_set_text.append(generator.generate(test_set_text[i]))
perturbed_train_set_text = train_set_text + perturbed_train_set_text
perturbed_train_set_labels = train_set_label + [
    label for label in train_set_label for i in range(NUM_OF_PERTURBATION)
]
perturbed_test_set_text = test_set_text + perturbed_test_set_text
perturbed_test_set_labels = test_set_label + [
    label for label in test_set_label for i in range(NUM_OF_PERTURBATION)
]

# Write the perturbed data to a csv file
with open(
    f"../perturbed_datasets/{DATA_SET}/{PERTURBATION}/{PERTURBATION}_{DISTRIBUTION}_{DENSITY}_train.csv",
    "w",
    newline="",
) as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"])
    writer.writerows(zip(perturbed_train_set_text, perturbed_train_set_labels))

with open(
    f"../perturbed_datasets/{DATA_SET}/{PERTURBATION}/{PERTURBATION}_{DISTRIBUTION}_{DENSITY}_test.csv",
    "w",
    newline="",
) as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"])
    writer.writerows(zip(perturbed_test_set_text, perturbed_test_set_labels))

# Tokenize the data and create tensors
tokenized_train_set = tokenizer(
    perturbed_train_set_text, truncation=True, padding=True, return_tensors="pt"
)
tokenized_train_set["labels"] = torch.LongTensor(perturbed_train_set_labels).clone()
tokenized_test_set = tokenizer(
    perturbed_test_set_text, truncation=True, padding=True, return_tensors="pt"
)
tokenized_test_set["labels"] = torch.LongTensor(perturbed_test_set_labels).clone()
tokenized_train_set = MyDataSet(tokenized_train_set)
tokenized_test_set = MyDataSet(tokenized_test_set)

# Create data loaders
train_loader = DataLoader(tokenized_train_set, shuffle=True, batch_size=128)
test_loader = DataLoader(tokenized_test_set, shuffle=True, batch_size=128)

# Initialise optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Start training
# These are for early stopping
i = 0
patience = 5
patience_count = 0
best_test_acc = 0
best_test_acc_round = 0
early_stopping = False
best_model_parameters = None
model.to(device)
while not early_stopping:
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    epoch_train_predictions = None
    epoch_train_labels = None
    num_of_train_batches = len(train_loader)
    for train_batch in train_loop:
        model.train()
        train_loss, train_predictions, train_accuracy, train_labels = train(
            train_batch, model, optimizer, device
        )
        train_loop.set_postfix(
            train_loss=train_loss.item(), train_accuracy=train_accuracy.item()
        )
        overall_train_loss += train_loss.item()
        train_loop.set_description(f"Round {i} train")
        if epoch_train_predictions is None:
            epoch_train_predictions = train_predictions
            epoch_train_labels = train_labels
        else:
            epoch_train_predictions = torch.cat(
                (epoch_train_predictions, train_predictions), dim=0
            )
            epoch_train_labels = torch.cat((epoch_train_labels, train_labels), dim=0)

    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    epoch_test_predictions = None
    epoch_test_labels = None
    num_of_test_batches = len(test_loader)
    for test_batch in test_loop:
        model.eval()
        test_loss, test_predictions, test_accuracy, test_labels = test(
            test_batch, model, device
        )
        test_loop.set_postfix(
            test_loss=test_loss.item(), test_accuracy=test_accuracy.item()
        )
        overall_test_loss += test_loss.item()
        test_loop.set_description(f"Round {i} test")

        if epoch_test_predictions is None:
            epoch_test_predictions = test_predictions
            epoch_test_labels = test_labels
        else:
            epoch_test_predictions = torch.cat(
                (epoch_test_predictions, test_predictions), dim=0
            )
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), dim=0)

    average_train_loss = overall_train_loss / num_of_train_batches
    epoch_train_accuracy = (
        torch.sum(torch.eq(epoch_train_predictions, epoch_train_labels))
        / epoch_train_labels.shape[0]
    )

    average_test_loss = overall_test_loss / num_of_test_batches
    epoch_test_accuracy = (
        torch.sum(torch.eq(epoch_test_predictions, epoch_test_labels))
        / epoch_test_labels.shape[0]
    )

    average_train_precision = metrics.precision_score(
        torch.flatten(epoch_train_labels).tolist(),
        torch.flatten(epoch_train_predictions).tolist(),
        average="macro",
    )
    average_train_recall = metrics.recall_score(
        torch.flatten(epoch_train_labels).tolist(),
        torch.flatten(epoch_train_predictions).tolist(),
        average="macro",
    )
    average_train_f1 = metrics.f1_score(
        torch.flatten(epoch_train_labels).tolist(),
        torch.flatten(epoch_train_predictions).tolist(),
        average="macro",
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
        f"Round {i} "
        f"train loss: {average_train_loss} "
        f"accuracy: {epoch_train_accuracy} "
        f"precision: {average_train_precision} "
        f"recall: {average_train_recall} f1: {average_train_f1}"
    )
    print(
        f"Round {i} "
        f"test loss: {average_test_loss} "
        f"accuracy: {epoch_test_accuracy} "
        f"precision: {average_test_precision} "
        f"recall: {average_test_recall} "
        f"f1: {average_test_f1}"
    )

    print(
        metrics.classification_report(
            torch.flatten(epoch_train_labels).tolist(),
            torch.flatten(epoch_train_predictions).tolist(),
        )
    )
    print(
        metrics.classification_report(
            torch.flatten(epoch_test_labels).tolist(),
            torch.flatten(epoch_test_predictions).tolist(),
        )
    )
    with open(RESULT_FILE, "a") as file:
        file.write(
            f"Round {i} "
            f"train loss: {average_train_loss} "
            f"accuracy: {epoch_train_accuracy} "
            f"precision: {average_train_precision} "
            f"recall: {average_train_recall} "
            f"f1: {average_train_f1}\n"
        )
        file.write(
            f"Round {i} "
            f"test loss: {average_test_loss} "
            f"accuracy: {epoch_test_accuracy} "
            f"precision: {average_test_precision} "
            f"recall: {average_test_recall} "
            f"f1: {average_test_f1}\n"
        )

    if epoch_test_accuracy <= best_test_acc:
        patience_count += 1
    else:
        best_test_acc = epoch_test_accuracy
        best_test_acc_round = i
        patience_count = 0
        best_model_parameters = [
            val.clone().detach().cpu().numpy() for _, val in model.state_dict().items()
        ]

    if patience_count >= patience:
        print(f"Early stopping at round {i}")
        print(f"Best test accuracy: {best_test_acc} at round {best_test_acc_round}")

        with open(RESULT_FILE, "a") as file:
            file.write(f"Early stopping at round {i}\n")
            file.write(
                f"Best test accuracy: {best_test_acc} at round {best_test_acc_round}\n"
            )
        # Save the best model parameters
        with open(
            f"../models/{DATA_SET}/{PERTURBATION}/"
            f"{MODEL_NAME}_{PERTURBATION}_{DISTRIBUTION}_{DENSITY}",
            "wb",
        ) as file:
            pickle.dump(best_model_parameters, file)
        early_stopping = True
    i += 1
