import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import module
from module import NeuralNetwork

trainDataSize = 20000
EPOCHS = 10
best_vloss = 1_000_000.


inputTenser = torch.rand(trainDataSize, 3)
resultData = []


for line in inputTenser:
    val = (line[0] * 2 - line[1] * 1 + line[2] * 2) / 3.0
    isBig = val > 0.5
    resultData.append([isBig])

resultTenser = torch.tensor(resultData, dtype=float)

inputTenserT = torch.rand(5, 3)
resultDataT = []
for line in inputTenserT:
    val = (line[0] * 2 + line[1] * 1 + line[2] * 2) / 3.0
    isBig = val > 0.5
    resultDataT.append([isBig])
resultTenserT = torch.tensor(resultDataT, dtype=float)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = NeuralNetwork().to(device)
print(model)


logits = model(inputTenser)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(inputTenser):
        # Every data instance is an input + label pair
        inputs = data
        labels = resultTenser[i]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i == (trainDataSize - 1):
            last_loss = running_loss / trainDataSize  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(inputTenser) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0



for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(inputTenserT):
        vinputs = vdata
        vlabels = resultTenserT[i]
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

model_path = 'model_{}_last'.format(timestamp)
torch.save(model.state_dict(), model_path)
