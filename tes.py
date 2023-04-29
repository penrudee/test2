import torch.nn as nn
from torchvision import models
from multi_task_model import * 
import os
import pandas as pd
from PIL import Image
import torch
import glob 
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import json
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np 

backbone = models.resnet18(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-1])  

num_classes_imprint = 57
num_classes_color = 11
num_classes_color2 = 11
num_classes_shape = 11

class MultiTaskPillModel(nn.Module):
    def __init__(self, backbone, num_imprints, num_primary_colors, num_secondary_colors, num_shapes):
        super(MultiTaskPillModel, self).__init__()
        self.backbone = backbone

        # Find the output size of the backbone model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.avgpool_output_size = dummy_output.view(dummy_output.size(0), -1).size(1)

        self.imprint_head = nn.Linear(self.avgpool_output_size, num_imprints)
        self.primary_color_head = nn.Linear(self.avgpool_output_size, num_primary_colors)
        self.secondary_color_head = nn.Linear(self.avgpool_output_size, num_secondary_colors)
        self.shape_head = nn.Linear(self.avgpool_output_size, num_shapes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        imprint_logits = self.imprint_head(x)
        primary_color_logits = self.primary_color_head(x)
        secondary_color_logits = self.secondary_color_head(x)
        shape_logits = self.shape_head(x)
        return imprint_logits, primary_color_logits, secondary_color_logits, shape_logits
    

multi_task_model = MultiTaskPillModel(backbone, 57, 11, 11, 11)


class PillDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        primary_colors = ["White", "Yellow", "Red","Pink","Orange","Blue","Purple","Green","Gray","Black","Transparent","Brown"]
        secondary_colors = ["White", "Yellow", "Red","Pink","Orange","Blue","Purple","Green","Gray","Black","Transparent","Brown"]
        shapes = ["Round", 
                  "Oval", 
                  "Oblong",
                  "Capsule",
                  "Triangle",
                  "Square",
                  "Pentagon",
                  "Hexagon",
                  "Octagon",
                  "Heart",
                  "Kidney",
                  "Other"]
        imprints = ["1", 
                    "60", 
                    "100",
                    "250",
                    "MASALAB",
                    "NL",
                    "PERAN",
                    "Line",
                    "MMT",
                    "ED",
                    "ADM",
                    "QUALIMED",
                    "C",
                    "MS,2",
                    "NL,10",
                    "SPS,PROPYL",
                    "ATC",
                    "PB",
                    "PHARMINAR,0,5",
                    "PC83",
                    "Line",
                    "GREATER",
                    "KB",
                    "None",
                    "SL",
                    "Line,50",
                    "4",
                    "U",
                    "ASIAN,5",
                    "Medicare",
                    "Plus",
                    "PML",
                    "ASIAN",
                    "Line,AC",
                    "TOC",
                    "FOB",
                    "20MG",
                    "AEH",
                    "SPS,006",
                    "CHINTA",
                    "Q",
                    "RANBAXY",
                    "Condrug",
                    "B2",
                    "PHARMINAR,1",
                    "42",
                    "Line,TO",
                    "BIOLAB",
                    "G,Line,10",
                    "Medicine,Supply",
                    "ASIAN,2",
                    "50",
                    "PatarLab",
                    "HK",
                    "BMP",
                    "Chinta",
                    "VIPERB12",
                     "E,32",
                      "ZC",
                      ]
        
        primary_colors_mapping={
            "White":0,
            "Yellow":1,
            "Red":2,
            "Pink":3,
            "Orange":4,
            "Blue":5,
            "Purple":6,
            "Green":7,
            "Gray":8,
            "Black":9,
            "Transparent":10,
            "Brown":11,
        }
        secondary_colors_mapping  ={
            "White":0,
            "Yellow":1,
            "Red":2,
            "Pink":3,
            "Orange":4,
            "Blue":5,
            "Purple":6,
            "Green":7,
            "Gray":8,
            "Black":9,
            "Transparent":10,
            "Brown":11,
        }
        shape_mapping={
            "Round":0, 
            "Oval":1, 
            "Oblong":2,
            "Capsule":3,
            "Triangle":4,
            "Square":5,
            "Pentagon":6,
            "Hexagon":7,
            "Octagon":8,
            "Heart":9,
            "Kidney":10,
            "Other":11,
        }
        imprints_mapping = {"1":0, 
                            "60":1, 
                            "100":2,
                            "250":3,
                            "MASALAB":4,
                            "NL":5,
                            "PERAN":6,
                            "Line":7,
                            "MMT":8,
                            "ED":9,
                            "ADM":10,
                            "QUALIMED":11,
                            "C":12,
                            "MS,2":13,
                            "NL,10":14,
                            "SPS,PROPYL":15,
                            "ATC":16,
                            "PB":17,
                            "PHARMINAR,0,5":18,
                            "PC83":19,
                            "Line":20,
                    "GREATER":21,
                    "KB":22,
                    "None":23,
                    "SL":24,
                    "Line,50":25,
                    "4":26,
                    "U":27,
                    "ASIAN,5":28,
                    "Medicare":29,
                    "Plus":30,
                    "PML":31,
                    "ASIAN":32,
                    "Line,AC":33,
                    "TOC":34,
                    "FOB":35,
                    "20MG":36,
                    "AEH":37,
                    "SPS,006":38,
                    "CHINTA":39,
                    "Q":40,
                    "RANBAXY":41,
                    "Condrug":42,
                    "B2":43,
                    "PHARMINAR,1":44,
                    "42":44,
                    "Line,TO":45,
                    "BIOLAB":46,
                    "G,Line,10":47,
                    "Medicine,Supply":48,
                    "ASIAN,2":49,
                    "50":50,
                    "PatarLab":51,
                    "HK":52,
                    "BMP":53,
                    "Chinta":54,
                    "VIPERB12":55,
                     "E,32":56,
                      "ZC":57,
                            }
        
        primary_color_to_index = {color: index for index, color in enumerate(primary_colors)}
        secondary_color_to_index = {color: index for index, color in enumerate(secondary_colors)}
        shape_to_index = {shape: index for index, shape in enumerate(shapes)}
        imprint_to_index = {imprint: index for index, imprint in enumerate(imprints)}
        # resize
        resize_transform = transforms.Resize((224, 224))
        
        image_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path).convert("RGB")
        image = resize_transform(image)
        image = to_tensor(image)
        

        label_primary_color = self.annotations.iloc[index, 0]
        label_secondary_color = self.annotations.iloc[index, 1]
        label_shape = self.annotations.iloc[index, 2]
        label_imprint = self.annotations.iloc[index, 3] #>>> version 1
        #label_imprint, label_primary_color, label_secondary_color, label_shape = self.annotations.iloc[index, 2:6] #>>>version 2 ,error

        ###################################################
        # Convert the labels to integers
        # label_primary_color = int(label_primary_color)
        # label_secondary_color = int(label_secondary_color)
        # label_shape = int(label_shape)
        label_primary_color = primary_colors_mapping.get(label_primary_color, 0)
        label_secondary_color = secondary_colors_mapping.get(label_secondary_color, 0)

        #################################################
         # Convert the string color labels to integer labels
        # label_primary_color = primary_colors_mapping.get(label_primary_color, 0)
        # label_secondary_color = secondary_colors_mapping.get(label_secondary_color, 0)
        # # label_trade_name = torch.tensor(self.annotations.iloc[index, 1])
        # if label_primary_color < 0 or label_primary_color >= num_classes_primary_color:
        #     label_primary_color = 0
        # if label_secondary_color < 0 or label_secondary_color >= num_classes_secondary_color:
        #     label_secondary_color = 0
        # if label_shape < 0 or label_shape >= num_classes_shape:
        #     label_shape = 0
        ################################################
       
        primary_color_index = primary_color_to_index.get(label_primary_color, -1)
        
        secondary_color_index = secondary_color_to_index.get(label_secondary_color, -1)
        shape_index = shape_to_index.get(label_shape, -1)
        imprint_index = imprint_to_index.get(label_imprint, -1)
        if label_imprint == -1 or label_primary_color == -1 or label_secondary_color == -1 or label_shape == -1:
          
          print(f"Invalid label found for {self.image_dir}: {label_imprint}, {label_primary_color}, {label_secondary_color}, {label_shape}")
        #   labels = torch.tensor([self.imprints_mapping[row['Imprint']], self.primary_colors_mapping[row['Color1']], self.secondary_colors_mapping[row['Color2']], self.shape_mapping[row['Shape']]], dtype=torch.long)

        return image, torch.tensor(primary_color_index, dtype=torch.long), torch.tensor(secondary_color_index, dtype=torch.long), torch.tensor(shape_index, dtype=torch.long), torch.tensor(imprint_index, dtype=torch.long)




data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# input_file = "/content/annotate.csv"
input_file=os.path.join(os.getcwd(),"dden27042566_csv.csv")
# output_file = "/content/multi_task_train.csv"
output_file = os.path.join(os.getcwd(),"multi_task_train.csv")

header = ["filename","Color1", "Color2", "Shape", "Imprint"]

with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    csv_reader = csv.DictReader(infile)
    csv_writer = csv.DictWriter(outfile, fieldnames=header)
    csv_writer.writeheader()

    for row in csv_reader:
        # region_shape_attributes = json.loads(row["region_shape_attributes"])
        # region_attributes = json.loads(row["region_attributes"])
        # print("region",region_shape_attributes)
        try:
            region_shape_attributes = json.loads(row["region_shape_attributes"])
            region_attributes = json.loads(row["region_attributes"])
            
        except json.JSONDecodeError:
            print(f"Skipping row with invalid JSON: {row}")
            continue
        pill_image = row["filename"] #ok แล้ว
        
        
        pill_color1 = region_attributes.get("Color1", "")
        
        pill_color2 = region_attributes.get("Color2", "")
        pill_shape = region_attributes.get("Shape", "")
        pill_imprint = region_attributes.get("Imprint", "")


        
        csv_writer.writerow({
            "filename": pill_image,
            "Color1": pill_color1,
            "Color2": pill_color2,
            "Shape": pill_shape,
            "Imprint": pill_imprint
        })
# Set paths
# C:\Users\DRUG INNOVATION CORP\dden_backend\all_img
# os.path.join(os.getcwd(),"pill_vall")
csv_path = os.path.join(os.getcwd(),"multi_task_train.csv")
images_folder = os.path.join(os.getcwd(),"all_img")

output_folder = os.getcwd()

# Read the CSV file
data = pd.read_csv(csv_path)

# Split the dataset
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the splitted data to separate CSV files
os.makedirs(output_folder, exist_ok=True)
train_data.to_csv(os.path.join(output_folder, "train_multi_task.csv"), index=False)
val_data.to_csv(os.path.join(output_folder, "val_multi_task.csv"), index=False)
test_data.to_csv(os.path.join(output_folder, "test_multi_task.csv"), index=False)

# Function to copy images to separate folders
def copy_images(data, src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for image_name in data["filename"]:
        src_path = os.path.join(src_folder, image_name)
        dst_path = os.path.join(dst_folder, image_name)
        shutil.copy(src_path, dst_path)

# Copy the images to separate folders
print("#"*20)
print("copy images section")

print("images folder",images_folder)
print("#"*20)
copy_images(train_data, images_folder, os.path.join(output_folder, "train_multi_task_images"))
copy_images(val_data, images_folder, os.path.join(output_folder, "val_multi_task_images"))
copy_images(test_data, images_folder, os.path.join(output_folder, "test_multi_task_images"))

train_dataset = PillDataset(csv_file="train_multi_task.csv", image_dir=os.path.join(os.getcwd(),"train_multi_task_images"), transform=data_transforms)
val_dataset = PillDataset(csv_file="val_multi_task.csv", image_dir=os.path.join(os.getcwd(),"val_multi_task_images"), transform=data_transforms)
test_dataset = PillDataset(csv_file="test_multi_task.csv", image_dir=os.path.join(os.getcwd(),"test_multi_task_images"), transform=data_transforms)


def custom_collate_fn(batch):
    images, imprint_labels, primary_color_labels, secondary_color_labels, shape_labels = zip(*batch)
    images = torch.stack(images)
    
    imprint_labels = torch.tensor(imprint_labels).view(-1, 1)
    primary_color_labels = torch.tensor(primary_color_labels).view(-1, 1)
    secondary_color_labels = torch.tensor(secondary_color_labels).view(-1, 1)
    shape_labels = torch.tensor(shape_labels).view(-1, 1)
    
    target = torch.cat((imprint_labels, primary_color_labels, secondary_color_labels, shape_labels), dim=1)
    
    return images, target


batch_size = 8

# version2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    
def train_multitask_model(model, train_loader, val_loader, epochs, device):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=-1) #เติมเข้ามาเพราะไปเจอ label=-1
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            imprint_labels = targets[:, 0].to(device)
            color_labels = targets[:, 1].to(device)
            color2_labels = targets[:, 2].to(device)
            shape_labels = targets[:, 3].to(device)

            optimizer.zero_grad()

            imprint_logits, color_logits, color2_logits, shape_logits = model(images)
            loss_imprint = criterion(imprint_logits, imprint_labels)
            loss_color = criterion(color_logits, color_labels)
            loss_color2 = criterion(color2_logits, color2_labels)
            loss_shape = criterion(shape_logits, shape_labels)
            loss = loss_imprint + loss_color + loss_color2 + loss_shape

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                imprint_labels = targets[:, 0].to(device)
                color_labels = targets[:, 1].to(device)
                color2_labels = targets[:, 2].to(device)
                shape_labels = targets[:, 3].to(device)

                imprint_logits, color_logits, color2_logits, shape_logits = model(images)
                loss_imprint = criterion(imprint_logits, imprint_labels)
                loss_color = criterion(color_logits, color_labels)
                loss_color2 = criterion(color2_logits, color2_labels)
                loss_shape = criterion(shape_logits, shape_labels)
                loss = loss_imprint + loss_color + loss_color2 + loss_shape

                running_val_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = running_val_loss / len(val_loader)
        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), "path_to_save_your_trained_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 25
train_multitask_model(multi_task_model, train_loader, val_loader, epochs, device)
