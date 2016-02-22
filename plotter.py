import matplotlib.pyplot as plt

def get(file_name):
    num = []
    with open(file_name, 'r') as f:
        for line in f:
            num.append(float(line.strip()))

    return num

valid_loss = get('valid_loss.txt')
train_loss = get('train_loss.txt')
valid_acc = get('valid_acc.txt')

plt.figure()
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss (NLL)')
valid_plot,  = plt.plot(valid_loss, 'b+', label='Valid loss')
train_plot, = plt.plot(train_loss, 'r+', label='Train loss')
plt.legend(handles=[valid_plot, train_plot])
plt.savefig('loss_plots.png')

plt.figure()
plt.title('Valid accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (percent)')
plt.plot(valid_acc, 'bo')
plt.savefig('acc_plot.png')

