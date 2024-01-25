import torch
import torchvision # should be inside the function since when it is called
# from the other module, it simply reads this function not above lines
from torch.utils.data.sampler import SubsetRandomSampler #shuffels automatically
import numpy as np
import matplotlib.pyplot as plt
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("  INSIDE LOADER.PY")

def data_loader():
    # print("  INSIDE data_loader()")
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,), inplace=False)]) # C=1
    #Composes several transforms together. transform the raw data in the defined format
    train_dataset = torchvision.datasets.MNIST("Data", download=True, train=True , transform=T)
    test_dataset = torchvision.datasets.MNIST("Data", download=True, train=False, transform=T)
    # print(len(train_indices)+len(test))
    
    #********************** splitting to validation and train **************************

    dataset_size = len(train_dataset) #60000
    shuffle_dataset=True
    validation_split = 0.2
    random_seed= 1
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    #********************************             DataLoader                     ***************************************

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler)#,  pin_memory=True, num_workers=1
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=valid_sampler)#, pin_memory=True, num_workers=1
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)#, pin_memory=True, num_workers=1
    #ValueError: sampler option is mutually exclusive with shuffle

    # shuffle for training, not for validation?????
    # intially data is loaded on cpu
    # train_feature, train_label = next(iter(train_loader)) # not iterate over ALL batches automatically
    # print('train_feature', train_feature.size(), train_feature.dim())
    # print('train_label', train_label.size(), train_label.dim())
    # print('train_feature.device', train_feature.device)#cpu
    # print('train_label.device', train_label.device)#cpu
    # # print(train_feature.shape)  #torch.Size([128, 1, 28, 28])
    # print(train_feature)
    # print(train_feature[1].all()==train_feature[10].all())
    # # print(train_label) # for only one batch
    # retrieve labels of ground truth
    
    # batch_idx=0
    # for batch_idx, (feature, labels) in enumerate(train_loader):
    #     print('train_feature.device', feature.device)#cpu
    #     print('train_label.device', labels.device)#cpu
    #     print(batch_idx)
    #     batch_idx += 1
    # #**********************       We can index Datasets manually like a list: traini[index]
    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 2, 2
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    #     img, label = train_dataset[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     # plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    #     plt.show()
    return train_loader, validation_loader, test_loader
# print("END of loader")    
data_loader()

#**************************           test 1 ***********************
# # Batch_size = 4 to show
# train_loader, validation_loader, test_loader = data_loader()
# Classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# def imshow(img):
#     plt.figure(figsize=(10,10))
#     img = img/2+0.5 # unnormalized
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
# # get some arndom training image
# dataiter = iter(train_loader)
# img, label = dataiter.next()

# imshow(torchvision.utils.make_grid(img))
# print(' '.join('%5s'% Classes[label[j]] for j in range(4)))
#***************************************************************************


# for batch_idx, (feature, labels) in enumerate(train_loader):
#     print('train_feature', feature.size())
#     print('train_label', labels.sshape[1])
#     print(batch_idx)
    # batch_idx += 1







