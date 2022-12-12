import torch
import matplotlib.pyplot as plt
import pandas as pd

def dataset_iterate(loader):
    for images, labels in loader:
        # print(images[0])
        # for image in images:
        #     print(image.shape)
        plt.imshow(torch.squeeze(images[0]))
        plt.show()
        break

def show_image(img_tensor):
    img_tensor = torch.squeeze(img_tensor).cpu()
    print('Value at (50,50):', img_tensor[50,50])
    plt.imshow(img_tensor)
    plt.show()
    return

def plot(path):
    df = pd.read_csv(path)

    df.plot(x="batch_num", y="val_acc", kind="line")

    plt.show()

plot("models/Circ_BonW_model1/Circ_BonW_model1_loss_data.csv")