from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils import train, test, MyDataSet
from sklearn import metrics

# Meta data
NUM_LABELS = 2
MODEL_NAME = "bert-base-uncased"
DATA_SET = "rotten_tomatoes"
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
# Tokenize the data and create tensors
tokenized_train_set = tokenizer(train_set["text"], truncation=True, padding=True, return_tensors="pt")
tokenized_train_set["labels"] = torch.LongTensor(train_set["label"]).clone()
tokenized_train_set = MyDataSet(tokenized_train_set)
# Combine val and test set and one test set
tokenized_test_set = tokenizer(val_set["text"] + test_set["text"], truncation=True, padding=True, return_tensors="pt")
tokenized_test_set["labels"] = torch.LongTensor(val_set["label"] + test_set["label"]).clone()
tokenized_test_set = MyDataSet(tokenized_test_set)
# Create data loaders
train_loader = DataLoader(tokenized_train_set, shuffle=True, batch_size=32)
test_loader = DataLoader(tokenized_test_set, shuffle=True, batch_size=32)
# Initialise optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
# Start training
model.to(device)
for i in range(EPOCH):
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
        train_loop.set_description(f"Epoch {i} train")
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
        test_loop.set_description(f"Epoch {i} test")

        if epoch_test_predictions is None:
            epoch_test_predictions = test_predictions
            epoch_test_labels = test_labels
        else:
            epoch_test_predictions = torch.cat((epoch_test_predictions, test_predictions), dim=0)
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), dim=0)

    average_train_loss = overall_train_loss / num_of_train_batches
    epoch_train_accuracy = torch.sum(torch.eq(epoch_train_predictions, epoch_train_labels)) / epoch_train_labels.shape[0]

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
        f"Epoch {i} train loss: {average_train_loss} accuracy: {epoch_train_accuracy} precision: {average_train_precision} recall: {average_train_recall} f1: {average_train_f1}")
    print(
        f"Epoch {i} test loss: {average_test_loss} accuracy: {epoch_test_accuracy} precision: {average_test_precision} recall: {average_test_recall} f1: {average_test_f1}")

    print(metrics.classification_report(torch.flatten(epoch_train_labels).tolist(),
                                        torch.flatten(epoch_train_predictions).tolist()))
    print(metrics.classification_report(torch.flatten(epoch_test_labels).tolist(),
                                        torch.flatten(epoch_test_predictions).tolist()))