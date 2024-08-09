import torch
import torchvision.utils as vutils
import argparse
import pickle
import os
from model import Generator 
from datasets import SuperCIFAR100 ,CIFAR_100_CLASS_MAP,tf,GeneratedDataset
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def parse_args():
    parser = argparse.ArgumentParser(description="Load and save model with custom configuration")
    parser.add_argument('--gennum', type=int, required=True, help='Generator number to customize filenames')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the generator model file')
    parser.add_argument('--pkl_paths', type=str, default=None,
                help='Optional: Comma-separated list of paths to PKL files containing images and labels.')
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
    for _, super_class_idx, _ in trainset:
        super_class_counts[super_class_idx] += 1

    # print("super_class_counts",super_class_counts)
    
    for idx, count in super_class_counts.items():
        print(f"{index_to_superclass[idx]}: {count} images")

    output_dir = f'generated_images_{args.gennum}'
    os.makedirs(output_dir, exist_ok=True)
    
    index_to_superclass = {i: super_class for i, super_class in enumerate(sorted(CIFAR_100_CLASS_MAP.keys()))}

    for idx, count in super_class_counts.items():
        print(f"{index_to_superclass[idx]}: {count} images")

    generated_images = []
    generated_labels = []
    for idx, total_images in super_class_counts.items():
        num_images_to_generate = int(total_images * 0.3)   
        print("idx: ",idx,"num_images_to_generate: ",num_images_to_generate)
    
        noise = torch.randn(num_images_to_generate, 100, device=device)
        labels = torch.full((num_images_to_generate,), idx, dtype=torch.long, device=device)
        with torch.no_grad():
            generated_images = gen(noise, labels)
        for i, image in enumerate(generated_images):
            vutils.save_image(image, os.path.join(output_dir, f'class_{index_to_superclass[idx]}_image_{i:04d}_{args.gennum}.png'), normalize=True)
            generated_images.append(image)
            generated_labels.append(idx)


    # Save the generated data to a specified directory with a custom file name
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    data_filename = f'generated_data_{args.gennum}.pkl'
    data_path = os.path.join(data_dir, data_filename)

    with open(data_path, 'wb') as f:
        pickle.dump((generated_images, generated_labels), f)

    print(f"Generated data saved to {data_path}")
    print("All images have been generated and saved.")
    

if __name__ == '__main__':
    main()