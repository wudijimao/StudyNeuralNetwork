import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

inputTenser = torch.rand(50, 3)
resultData = []


for line in inputTenser:
    val = (line[0] + line[1] + line[2]) / 3.0
    resultData.append([val])

resultTenser = torch.tensor(resultData, dtype=float)

inputTenserT = torch.rand(5, 3)
resultDataT = []
for line in inputTenserT:
    val = (line[0] + line[1] + line[2]) / 3.0
    resultDataT.append([val])
resultTenserT = torch.tensor(resultDataT, dtype=float)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=3, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
# model = NeuralNetwork().to(device)
# print(model)


# logits = model(inputTenser)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")


# print(f"Model structure: {model}\n\n")

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# loss_fn = torch.nn.L1Loss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# def train_one_epoch(epoch_index, tb_writer):
#     running_loss = 0.
#     last_loss = 0.

#     # Here, we use enumerate(training_loader) instead of
#     # iter(training_loader) so that we can track the batch
#     # index and do some intra-epoch reporting
#     for i, data in enumerate(inputTenser):
#         # Every data instance is an input + label pair
#         inputs = data
#         labels = resultTenser[i]

#         # Zero your gradients for every batch!
#         optimizer.zero_grad()

#         # Make predictions for this batch
#         outputs = model(inputs)

#         # Compute the loss and its gradients
#         loss = loss_fn(outputs, labels)
#         loss.backward()

#         # Adjust learning weights
#         optimizer.step()

#         # Gather data and report
#         running_loss += loss.item()
#         if i % 50 == 49:
#             last_loss = running_loss / 50  # loss per batch
#             print('  batch {} loss: {}'.format(i + 1, last_loss))
#             tb_x = epoch_index * len(inputTenser) + i + 1
#             tb_writer.add_scalar('Loss/train', last_loss, tb_x)
#             running_loss = 0.

#     return last_loss


# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
# epoch_number = 0

# EPOCHS = 200

# best_vloss = 1_000_000.

# for epoch in range(EPOCHS):
#     print('EPOCH {}:'.format(epoch_number + 1))

#     # Make sure gradient tracking is on, and do a pass over the data
#     model.train(True)
#     avg_loss = train_one_epoch(epoch_number, writer)

#     # We don't need gradients on to do reporting
#     model.train(False)

#     running_vloss = 0.0
#     for i, vdata in enumerate(inputTenserT):
#         vinputs = vdata
#         vlabels = resultTenserT[i]
#         voutputs = model(vinputs)
#         vloss = loss_fn(voutputs, vlabels)
#         running_vloss += vloss

#     avg_vloss = running_vloss / (i + 1)
#     print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

#     # Log the running loss averaged per batch
#     # for both training and validation
#     writer.add_scalars('Training vs. Validation Loss',
#                        {'Training': avg_loss, 'Validation': avg_vloss},
#                        epoch_number + 1)
#     writer.flush()

#     # Track best performance, and save the model's state
#     if avg_vloss < best_vloss:
#         best_vloss = avg_vloss
#         model_path = 'model_{}_{}'.format(timestamp, epoch_number)
#         torch.save(model.state_dict(), model_path)

#     epoch_number += 1

# To load a saved version of the model:
saved_model = NeuralNetwork()
saved_model.load_state_dict(torch.load("model_20220627_122006_21"))
val0 = saved_model(torch.tensor(
    [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.49, 0.5, 0.5], [0.2, 0.2, 1.2], [0.49, 0.49, 0.49]], dtype=torch.float32))
val1 = saved_model(inputTenserT)
print(val0)
print(inputTenserT)
print(val1)
