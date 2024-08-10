import torch
import torchvision.utils as vutils
import argparse
import pickle
import os
from model import Generator 
from datasets import SuperCIFAR100 ,CIFAR_100_CLASS_MAP,tf,GeneratedDataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def parse_args():
    parser = argparse.ArgumentParser(description="Load and save model with custom configuration")
    parser.add_argument('--gennum', type=int, required=True, help='Generator number to customize filenames')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the generator model file')
    
    return parser.parse_args()


def load_model(model_path):
 
    model = Generator().to(device)  # 替换为你的实际模型类
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded generator model from {model_path}")
    model.eval()  # 将模型设置为评估模式
    return model

def main():
    args = parse_args()
    model_dir = args.model_path
    gen_model_path = os.path.join(model_dir, f'gen_{args.gennum}.pth')
  
    gen = load_model(gen_model_path)
  
    trainset = SuperCIFAR100(root='./data', train=True, download=False, transform=tf)

    # 统计每个大类的图片数量
    super_class_counts = {i: 0 for i in range(len(CIFAR_100_CLASS_MAP))}  # 初始化所有大类的计数为0
    # for _, super_class_idx, _ in trainset:
    #     super_class_counts[super_class_idx] += 1

  

    output_dir = f'data/generated_images_{args.gennum}'
    os.makedirs(output_dir, exist_ok=True)
    
    index_to_superclass = {i: super_class for i, super_class in enumerate(sorted(CIFAR_100_CLASS_MAP.keys()))}

    if args.gennum == 0:
        for _, super_class_idx,_ in trainset:
            super_class_counts[super_class_idx] += 1
        print("super_class_counts",super_class_counts)
    
    else:
        previous_gen_path = f'data/generated_images_{args.gennum - 1}'
         
        for class_folder in os.listdir(previous_gen_path):
            class_idx = int(class_folder.split('_')[-1])
            class_path = os.path.join(previous_gen_path, class_folder)
            image_count = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])
            super_class_counts[class_idx] += image_count

    print("super_class_counts", super_class_counts)


    generated_images = []
    generated_labels = []
    for idx, total_images in super_class_counts.items():
        num_images_to_generate = int(total_images * 0.3)   
        print("idx: ",idx,"num_images_to_generate: ",num_images_to_generate)
    
        noise = torch.randn(num_images_to_generate, 100, device=device)
        labels = torch.full((num_images_to_generate,), idx, dtype=torch.long, device=device)
        with torch.no_grad():
            new_generated_images = gen(noise, labels)
        # creat a new directory for each class
        os.makedirs(os.path.join(output_dir, f'class_{idx}'), exist_ok=True)

        for i, image in enumerate(new_generated_images):
            save_path = os.path.join(output_dir, f'class_{idx}', f'image_{i:04d}_{args.gennum}.png')
            # vutils.save_image(image, os.path.join(output_dir, f'class_{index_to_superclass[idx]}_image_{i:04d}_{args.gennum}.png'), normalize=True)
            vutils.save_image(image, save_path, normalize=True)

    # Save the generated data to a specified directory with a custom file name
    

    print(f"Generated data saved to {output_dir}")
    print("All images have been generated and saved.")
    

if __name__ == '__main__':
    main()
