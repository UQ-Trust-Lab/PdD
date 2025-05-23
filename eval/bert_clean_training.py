import pickle

import torch
from datasets import load_dataset
from sklearn import metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast

from eval.utils import *

# Metadata
NUM_LABELS = 2
MODEL_NAME = "bert-base-uncased"
DATA_SET = "rotten_tomatoes"
RESULT_FILE = f"../../results/{DATA_SET}/{MODEL_NAME}_clean_training.txt"

# Initialise model and tokenizer from metadata
# cuda:1 is just for when cuda:0 is taken
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
# Load the data
dataset = load_dataset(DATA_SET)
train_set = dataset["train"]
val_set = dataset["validation"]
test_set = dataset["test"]
# Tokenize the data and create tensors
tokenized_train_set = tokenizer(
    train_set["text"], truncation=True, padding=True, return_tensors="pt"
)
tokenized_train_set["labels"] = torch.LongTensor(train_set["label"]).clone()
tokenized_train_set = MyDataSet(tokenized_train_set)
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
        train_loop.set_description(f"Epoch {i} train")
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
        test_loop.set_description(f"Epoch {i} test")

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
        f"Epoch {i} "
        f"train loss: {average_train_loss} "
        f"accuracy: {epoch_train_accuracy} "
        f"precision: {average_train_precision} "
        f"recall: {average_train_recall} "
        f"f1: {average_train_f1}"
    )
    print(
        f"Epoch {i} "
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
        with open(f"../../models/{DATA_SET}/{MODEL_NAME}_clean_training", "wb") as file:
            pickle.dump(best_model_parameters, file)
        early_stopping = True
    i += 1
