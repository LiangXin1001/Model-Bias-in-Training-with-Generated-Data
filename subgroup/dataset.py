from robustness.tools.breeds_helpers import make_living17
from torch.utils.data import Dataset
import torchvision
import torch




class Living17Dataset(Dataset):
    def __init__(self, data_dir, data_base, split='', transform=None, train=True):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.dataset_info = make_living17(data_dir, split=split)
        superclasses, subclass_split, label_map = self.dataset_info
        num_superclasses = len(subclass_split[0])
        self.num_superclasses = num_superclasses

        subclasses = []
        all_subclasses = []
        for i in range(num_superclasses):
            subclasses.append(subclass_split[0][i])
            subclasses[i].extend(subclass_split[1][i])
        
        for i in range(num_superclasses):
            all_subclasses.extend(subclasses[i])

        self.ori2label = {}
        self.label2ori = {}
        for subclass_index in range(len(all_subclasses)):
            self.ori2label[all_subclasses[subclass_index]] = subclass_index
            self.label2ori[subclass_index] = all_subclasses[subclass_index]

        self.subclasses = subclasses
        self.label_map = label_map
        self.sub2super = {}
        for i in range(num_superclasses):
            for subclass in subclasses[i]:
                self.sub2super[subclass] = i
        
        if train:
            self.data = torchvision.datasets.ImageNet(data_base, split='train', transform=transform)
        else:
            self.data = torchvision.datasets.ImageNet(data_base, split='val', transform=transform)

        selected_indices = []
        for data_i in range(len(self.data)):
            if self.data.targets[data_i] in all_subclasses:
                selected_indices.append(data_i)
        self.selected_indices = selected_indices
        self.sub_data = torch.utils.data.Subset(self.data, selected_indices)


    def __len__(self):
        return len(self.sub_data)
    
    def __getitem__(self, idx):
        img, label = self.sub_data[idx]
        new_label = self.ori2label[label]
        super_label = self.sub2super[label]
        # import pdb; pdb.set_trace()
        return img, new_label, super_label
    

    

if __name__ == "__main__":
    info_dir = "/home/v-zelzhang/unlearn/week4/info_dir"
    imagenet_path = "/home/v-zelzhang/unlearn/data/imagenet"
    dataset = Living17Dataset(info_dir, imagenet_path,train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (img, label, super_label) in enumerate(dataloader):
        print(label)
        print(super_label)
        break
