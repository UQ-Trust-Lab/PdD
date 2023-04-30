import pickle
import csv
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
from new_version.perturbation import Generator, DELETION_DICT, HOMOGLYPHS_DICT, INVISIBLE_UNICODE_CHARS, KEYBOARD_TYPO_ADVANCED_DICT
from collections import defaultdict
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils import train, test, MyDataSet, set_parameters
from sklearn import metrics
from argparse import ArgumentParser

parser = ArgumentParser(description='PDD TESTER')
parser.add_argument('--perturbation', default="deletion", type=str,
                    help='perturbation method')
parser.add_argument('--distribution', default="uniform", type=str,
                    help='distribution of perturbation')
parser.add_argument('--density', default=0.05, type=float,
                    help='density of perturbation')

args = parser.parse_args()

# Meta data
NUM_LABELS = 2
MODEL_NAME = "bert-base-uncased"
DATA_SET = "rotten_tomatoes"
PERTURBATION = args.perturbation
DISTRIBUTION = args.distribution
DENSITY = args.density
NUM_OF_PERTURBATION = 10
RESULT_FILE = f"../../results/{DATA_SET}/{MODEL_NAME}_{PERTURBATION}_{DISTRIBUTION}_{DENSITY}_robust_training.txt"
CLEAN_MODEL = f"../../models/{DATA_SET}/{MODEL_NAME}_clean_training"
with open(RESULT_FILE, "a") as file:
    file.write(f"Model: {MODEL_NAME}\n")
    file.write(f"Dataset: {DATA_SET}\n")
    file.write(f"Perturbation: {PERTURBATION}\n")
    file.write(f"Distribution: {DISTRIBUTION}\n")
    file.write(f"Density: {DENSITY}\n")
    file.write(f"Number of perturbation: {NUM_OF_PERTURBATION}\n")

diversity_dict = defaultdict(lambda: [' '])

PERTURBATION_DICT = {
    "deletion": DELETION_DICT,
    "typo": KEYBOARD_TYPO_ADVANCED_DICT,
    "homoglyphs": HOMOGLYPHS_DICT,
    "invisible": INVISIBLE_UNICODE_CHARS
}

diversity_dict.update(PERTURBATION_DICT[PERTURBATION])

# Initialise model and tokenizer from meta data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
clean_model_params = pickle.load(open(CLEAN_MODEL, "rb"))
model = set_parameters(model, clean_model_params)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# Load the data
dataset = load_dataset(DATA_SET)
train_set = dataset["train"]
val_set = dataset["validation"]
test_set = dataset["test"]

# Initialise the perturbation generator
generator = Generator(DISTRIBUTION, DENSITY, diversity_dict)
train_set_text = train_set["text"]
train_set_label = train_set["label"]
test_set_text = val_set["text"] + test_set["text"]
test_set_label = val_set["label"] + test_set["label"]

# Add perturbation to the data
perturbed_train_set_text = []
for i in range(len(train_set_text)):
    for j in range(NUM_OF_PERTURBATION):
        generator = Generator(DISTRIBUTION, DENSITY, diversity_dict)
        perturbed_train_set_text.append(generator.generate(train_set_text[i]))
perturbed_test_set_text = []
for i in range(len(test_set_text)):
    for j in range(NUM_OF_PERTURBATION):
        generator = Generator(DISTRIBUTION, DENSITY, diversity_dict)
        perturbed_test_set_text.append(generator.generate(test_set_text[i]))
perturbed_train_set_text = train_set_text + perturbed_train_set_text
perturbed_train_set_labels = train_set_label + [label for label in train_set_label for i in range(NUM_OF_PERTURBATION)]
perturbed_test_set_text = test_set_text + perturbed_test_set_text
perturbed_test_set_labels = test_set_label + [label for label in test_set_label for i in range(NUM_OF_PERTURBATION)]

# Write the perturbed data to a csv file
with open(f"../../perturbed_datasets/{DATA_SET}/{PERTURBATION}_{DISTRIBUTION}_{DENSITY}_train.csv", "w",
          newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"])
    writer.writerows(zip(perturbed_train_set_text, perturbed_train_set_labels))

with open(f"../../perturbed_datasets/{DATA_SET}/{PERTURBATION}_{DISTRIBUTION}_{DENSITY}_test.csv", "w",
          newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"])
    writer.writerows(zip(perturbed_test_set_text, perturbed_test_set_labels))

# Tokenize the data and create tensors
tokenized_train_set = tokenizer(perturbed_train_set_text, truncation=True, padding=True, return_tensors="pt")
tokenized_train_set["labels"] = torch.LongTensor(perturbed_train_set_labels).clone()
tokenized_test_set = tokenizer(perturbed_test_set_text, truncation=True, padding=True, return_tensors="pt")
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
        train_loss, train_predictions, train_accuracy, train_labels = train(train_batch, model, optimizer, device)
        train_loop.set_postfix(train_loss=train_loss.item(), train_accuracy=train_accuracy.item())
        overall_train_loss += train_loss.item()
        train_loop.set_description(f"Round {i} train")
        if epoch_train_predictions is None:
            epoch_train_predictions = train_predictions
            epoch_train_labels = train_labels
        else:
            epoch_train_predictions = torch.cat((epoch_train_predictions, train_predictions), dim=0)
            epoch_train_labels = torch.cat((epoch_train_labels, train_labels), dim=0)

    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    epoch_test_predictions = None
    epoch_test_labels = None
    num_of_test_batches = len(test_loader)
    for test_batch in test_loop:
        model.eval()
        test_loss, test_predictions, test_accuracy, test_labels = test(test_batch, model, device)
        test_loop.set_postfix(test_loss=test_loss.item(), test_accuracy=test_accuracy.item())
        overall_test_loss += test_loss.item()
        test_loop.set_description(f"Round {i} test")

        if epoch_test_predictions is None:
            epoch_test_predictions = test_predictions
            epoch_test_labels = test_labels
        else:
            epoch_test_predictions = torch.cat((epoch_test_predictions, test_predictions), dim=0)
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), dim=0)

    average_train_loss = overall_train_loss / num_of_train_batches
    epoch_train_accuracy = torch.sum(torch.eq(epoch_train_predictions, epoch_train_labels)) / epoch_train_labels.shape[
        0]

    average_test_loss = overall_test_loss / num_of_test_batches
    epoch_test_accuracy = torch.sum(torch.eq(epoch_test_predictions, epoch_test_labels)) / epoch_test_labels.shape[0]

    average_train_precision = metrics.precision_score(torch.flatten(epoch_train_labels).tolist(),
                                                      torch.flatten(epoch_train_predictions).tolist(), average="macro")
    average_train_recall = metrics.recall_score(torch.flatten(epoch_train_labels).tolist(),
                                                torch.flatten(epoch_train_predictions).tolist(), average="macro")
    average_train_f1 = metrics.f1_score(torch.flatten(epoch_train_labels).tolist(),
                                        torch.flatten(epoch_train_predictions).tolist(), average="macro")

    average_test_precision = metrics.precision_score(torch.flatten(epoch_test_labels).tolist(),
                                                     torch.flatten(epoch_test_predictions).tolist(), average="macro")
    average_test_recall = metrics.recall_score(torch.flatten(epoch_test_labels).tolist(),
                                               torch.flatten(epoch_test_predictions).tolist(), average="macro")
    average_test_f1 = metrics.f1_score(torch.flatten(epoch_test_labels).tolist(),
                                       torch.flatten(epoch_test_predictions).tolist(), average="macro")

    print(
        f"Round {i} train loss: {average_train_loss} accuracy: {epoch_train_accuracy} precision: {average_train_precision} recall: {average_train_recall} f1: {average_train_f1}")
    print(
        f"Round {i} test loss: {average_test_loss} accuracy: {epoch_test_accuracy} precision: {average_test_precision} recall: {average_test_recall} f1: {average_test_f1}")

    print(metrics.classification_report(torch.flatten(epoch_train_labels).tolist(),
                                        torch.flatten(epoch_train_predictions).tolist()))
    print(metrics.classification_report(torch.flatten(epoch_test_labels).tolist(),
                                        torch.flatten(epoch_test_predictions).tolist()))
    with open(RESULT_FILE, "a") as file:
        file.write(
            f"Round {i} train loss: {average_train_loss} accuracy: {epoch_train_accuracy} precision: {average_train_precision} recall: {average_train_recall} f1: {average_train_f1}\n")
        file.write(
            f"Round {i} test loss: {average_test_loss} accuracy: {epoch_test_accuracy} precision: {average_test_precision} recall: {average_test_recall} f1: {average_test_f1}\n")

    if epoch_test_accuracy <= best_test_acc:
        patience_count += 1
    else:
        best_test_acc = epoch_test_accuracy
        best_test_acc_round = i
        patience_count = 0
        best_model_parameters = [val.clone().detach().cpu().numpy() for _, val in model.state_dict().items()]

    if patience_count >= patience:
        print(f"Early stopping at round {i}")
        print(f"Best test accuracy: {best_test_acc} at round {best_test_acc_round}")

        with open(RESULT_FILE, "a") as file:
            file.write(f"Early stopping at round {i}\n")
            file.write(f"Best test accuracy: {best_test_acc} at round {best_test_acc_round}\n")
        # Save the best model parameters
        with open(f"../../models/{DATA_SET}/{MODEL_NAME}_{PERTURBATION}_{DISTRIBUTION}_{DENSITY}", "wb") as file:
            pickle.dump(best_model_parameters, file)
        early_stopping = True
    i += 1