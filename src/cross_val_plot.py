# -*- coding:utf-8 -*-

import re
import matplotlib.pyplot as plt

lgn = 'cross_val'
log_file = open('logs/{}.log'.format(lgn))
s = log_file.readlines()
log_file.close()

train_acc = []
val_acc = []

for line in s:
    result = re.findall(
        r"^.*Validation Index[ *]\[(\d*)\].*(Training|Validation).*Accuracy:[ *]([0-9,.]+)", line)
    if len(result) > 0:
        result = result[0]
        if result[1] == 'Training':
            train_acc.append(float(result[2]))
        elif result[1] == 'Validation':
            val_acc.append(float(result[2]))


trainx = range(len(train_acc))
valx = range(len(val_acc))

print 'train_acc:', train_acc
print 'validation_acc:', val_acc

plt.title('Training&Validation Loss&Accuracy')

plt.subplot(2, 1, 1)
plt.plot(trainx, train_acc, marker='.', color='r',
         label='Training Phase Accuracy')
plt.xlabel('val_index')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(valx, val_acc, marker='o', color='r',
         label='Validation Phase Accuracy')
plt.xlabel('val_index')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
