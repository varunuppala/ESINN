import torch
import matplotlib.pyplot as plt

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