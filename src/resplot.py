# -*- coding:utf-8 -*-

import re
import matplotlib.pyplot as plt

lgn = 'resnet'
log_file = open('logs/{}.log'.format(lgn))
s = log_file.readlines()
log_file.close()

train_acc = []
train_loss = []
val_acc = []
val_loss = []

for line in s:
    result = re.findall(
        r"^.*Epoch[ *]\[(\d*)\].*(Training|Validation).*Loss:[ *]([0-9,.]+).*Accuracy:[ *]([0-9,.]+)", line)
    if len(result) > 0:
        result = result[0]
        if result[1] == 'Training':
            train_acc.append(float(result[3]))
            train_loss.append(float(result[2]))
        elif result[1] == 'Validation':
            val_acc.append(float(result[3]))
            val_loss.append(float(result[2]))


trainx = range(len(train_acc))
valx = range(len(val_acc))

plt.title('Training&Validation Loss&Accuracy')

plt.subplot(2, 2, 1)
plt.plot(trainx, train_acc, marker='.', color='r',
         label='Training Phase Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(2, 2, 2)
plt.plot(valx, val_acc, marker='.', color='r',
         label='Validation Phase Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(2, 2, 3)
plt.plot(trainx, train_loss, marker='.', color='b',
         label='Training Phase Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(2, 2, 4)
plt.plot(valx, val_loss, marker='.', color='b',
         label='Validation Phase Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()
