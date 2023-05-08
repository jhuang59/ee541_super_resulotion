
class Div2K(Dataset):
  def __init__(self, input_folder, output_folder, transform_input, transform_output):
    self.input_folder = input_folder
    self.output_folder = output_folder
    self.transform_input = transform_input
    self.transform_output = transform_output
    self.input_images = sorted([file for file in os.listdir(input_folder) if file.endswith('.png')])
    self.output_images = sorted([file for file in os.listdir(output_folder) if file.endswith('.png')])
    
  def __len__(self):
    return len(self.input_images)
  def __getitem__(self,index):
    input_image_path = os.path.join(self.input_folder, self.input_images[index])
    output_image_path = os.path.join(self.output_folder, self.output_images[index])
    input_image = Image.open(input_image_path).convert('RGB')
    output_image = Image.open(output_image_path).convert('RGB')

    input_image = self.transform_input(input_image)
    output_image = self.transform_output(output_image)
    return input_image, output_image

train_dataset = Div2K(train_lr, train_hr, transform_input=transform['train_input'],transform_output=transform['train_output'])
train_dataset_rotate = Div2K(train_lr, train_hr, transform_input=transform['train_input_rotate'],transform_output=transform['train_output_rotate'])
train_dataset_flip = Div2K(train_lr, train_hr, transform_input=transform['train_input_flip'],transform_output=transform['train_output_flip'])
test_dataset = Div2K(test_lr, test_hr, transform_input = transform['val_test_input'],transform_output=transform['val_test_output'])
val_dataset = Div2K(val_lr, val_hr, transform_input = transform['val_test_input'],transform_output=transform['val_test_output'])

temp = torch.utils.data.ConcatDataset([train_dataset, train_dataset_rotate])
train_dataset_combined = torch.utils.data.ConcatDataset([temp, train_dataset_flip])

batch_size = 24

train_dataloader_rotate = DataLoader(train_dataset_rotate, batch_size=batch_size, shuffle=True, num_workers=2)
train_dataloader_flip = DataLoader(train_dataset_flip, batch_size=batch_size, shuffle=True, num_workers=2)
train_dataloader_combined = DataLoader(train_dataset_combined, batch_size=batch_size, shuffle=True, num_workers=2)
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)