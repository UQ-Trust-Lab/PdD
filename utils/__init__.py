"""
Some utility functions and classes for model training and evaluation.
"""
from collections import OrderedDict

import torch


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def train(batch, model, optimizer, device):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                          labels=labels)
    train_loss = train_outputs.loss
    train_loss.backward()
    optimizer.step()
    predictions = torch.argmax(train_outputs.logits, dim=-1)
    accuracy = torch.sum(torch.eq(predictions, labels)) / labels.shape[0]
    return train_loss, predictions, accuracy, labels


def validate(batch, model, device):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    val_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                        labels=labels)
    val_loss = val_outputs.loss
    predictions = torch.argmax(val_outputs.logits, dim=-1)
    accuracy = torch.sum(torch.eq(predictions, labels)) / labels.shape[0]
    return val_loss, predictions, accuracy, labels


def test(batch, model, device):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         labels=labels)
    test_loss = test_outputs.loss
    predictions = torch.argmax(test_outputs.logits, dim=-1)
    accuracy = torch.sum(torch.eq(predictions, labels)) / labels.shape[0]
    return test_loss, predictions, accuracy, labels


def set_parameters(model, parameters):
    """
    Set the parameters of a model.
    Args:
        model: A neural network models with parameters.
        parameters: A list of parameters for the model.

    Returns:
        The model with the new parameters.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model
