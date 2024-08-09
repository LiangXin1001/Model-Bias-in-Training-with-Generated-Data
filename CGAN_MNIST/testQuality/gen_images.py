import torch
import os
import torchvision.utils as vutils
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):

    def __init__(self, image_size: int = 28, channels: int = 3, num_classes: int = 10) -> None:
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(100 + num_classes, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(1024, channels * image_size * image_size),
            nn.Tanh()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

 

def generate_images_and_save(model, num_images_per_class=1000, num_classes=10):
    model.eval() 

    base_src_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/weights/weights_gen'
    base_dest_path = '/local/scratch/hcui25/Project/xin/CS/GAN/testQuality/gen'
    src_dest_pairs = {}
 
    for i in range(1, 10):  
        src_path = f"{base_src_path}{i}/GAN-last.pth"
        dest_path = f"{base_dest_path}{i - 1}"
        src_dest_pairs[src_path] = dest_path

    for src_path, dest_path in src_dest_pairs.items():
        os.makedirs(dest_path, exist_ok=True)  
 
        checkpoint = torch.load(src_path)
        model.load_state_dict(checkpoint['state_dict'])

        for class_idx in range(num_classes):
            conditional = torch.full((num_images_per_class,), class_idx, dtype=torch.long)
            noise = torch.randn(num_images_per_class, 100)
 
            if device is not None:
                noise = noise.to(device)
                conditional = conditional.to(device)

            with torch.no_grad():
                images = model(noise, conditional)
 
            for j, image in enumerate(images):
                image_filename = f"class_{class_idx}_image_{j:04d}.png"
                save_path = os.path.join(dest_path, image_filename)

                vutils.save_image(image, save_path, normalize=True)

            print(f"Generated and saved {num_images_per_class} images for class {class_idx} in gen{i - 1}")
 

model = Generator()

generate_images_and_save(model, num_images_per_class=1000, num_classes=10) 
print("图片生成并保存完成。")
 