#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, datasets, transforms

import sagemaker

import os
import argparse
import time
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

device_type = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device type is '{device_type}'")
device = torch.device(device_type)

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    running_loss = 0
    running_correct = 0
    test_set_size = len(test_loader.dataset)

    model.eval() # set model to evaluation mode

    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            #preds = outputs.argmax(dim=1, keepdim=True) # get the index of the max probability
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)

            # stats
            loss_count = loss.item() * inputs.size(0)
            running_loss += loss_count
            #correct_count = preds.eq(targets.view_as(preds)).sum(),item()
            correct_count = torch.sum(preds == targets.data)
            running_correct += correct_count

        loss_ratio = running_loss / test_set_size
        accuracy = running_correct / test_set_size

        print(f"\nTest set: Average loss: {loss_ratio:.4f}, Accuracy: {running_correct}/{test_set_size} = {accuracy:.4f}\n")

def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # decay LR by a factor of 0.1 every 7 epochs 

    now_str = str(time.time())
    best_model_params_file_name = now_str + '_best_model_params.pt'
    #best_model_params_path = './' + best_model_params_file_name
    #torch.save(model.state_dict(), best_model_params_path) # save model params of the existing model
    best_model = model
    best_accuracy = 0.0
    has_improved = True

    for epoch in range(epochs):

        for type in ['train', 'val']:

            if type == 'train':
                model.train() # set model to train mode
                data_loader = train_loader
                #torch.set_grad_enabled(True)
            else:
                model.eval() # set model to evaluate mode
                data_loader = val_loader
                #torch.set_grad_enabled(False)

            image_count = len(data_loader.dataset)
            batch_count = len(data_loader)

            running_loss = 0
            running_correct = 0

            for batch_id, (inputs, targets) in enumerate(data_loader):

                inputs = inputs.to(device)
                targets = targets.to(device)

                if type == 'train':
                    optimizer.zero_grad()

                outputs = model(inputs)
                #preds = outputs.argmax(dim=1, keepdim=True) # get the index of the max probability
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, targets)

                # stats
                loss_count = loss.item() * inputs.size(0)
                running_loss += loss_count
                #correct_count = preds.eq(targets.view_as(preds)).sum(),item()
                correct_count = torch.sum(preds == targets.data)
                running_correct += correct_count

                if type == 'train':
                    loss.backward()
                    optimizer.step()

                batch_accuracy = correct_count / len(inputs)
                print(f"Pass {epoch}/{epochs} {type} Batch {batch_id}/{batch_count} {len(inputs)}/{inputs.size(0)} - Loss count: {loss.item()}/{loss_count}, Accuracy: {correct_count}/{batch_accuracy:.4f}")

            """
            if type == 'train':
                scheduler.step()
            """

            loss_ratio = running_loss / image_count
            accuracy = running_correct / image_count
            print(f"Pass {epoch}/{epochs} {type} - Average loss: {loss_ratio:.4f}, Accuracy: {accuracy:.4f}")

            if type == 'val':
                if accuracy > best_accuracy:
                    best_model = model
                    best_accuracy = accuracy
                    # save model params 
                    #torch.save(model.state_dict(), best_model_params_path)
                    has_improved = True
                else:
                    # should we stop if the accuracy is not better than the previous epoch?
                    print(f'Accuracy has not improved from the previous epoch {epoch-1}, we will stop!')
                    has_improved = False

        if has_improved == False:
            break

    print('We are done with training and validation, and will return the best model!')

    # load from the best model params file
    #model.load_state_dict(torch.load(best_model_params_path))

    return best_model, best_model_params_file_name

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # we will start with a simpler 18-layer network
    #model = models.resnet18(weights='IMAGENET1K_V1')
    model = models.resnet18(pretrained=True)

    # we will start with the fixed feature extractor scenario, which replaces the last FC layer with a new
    # one with random weights, and only train that layer, this will take shorter time to train    
    for param in model.parameters():
        param.requires_grad = False

    in_feature_count = model.fc.in_features
    model.fc = nn.Linear(in_feature_count, 133)

    model = model.to(device)

    print('Created a model fined tuned from RESNET18')

    return model

def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    # code from MNIST example
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    """
    # code from PyTorch ants bees fine tune example
    # for training data: data augmentation + normalize
    # for validation data: just normalize
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(data_dir + '/train', data_transforms['train']),
        'val': datasets.ImageFolder(data_dir + '/valid', data_transforms['val']),
        'test': datasets.ImageFolder(data_dir + '/test', data_transforms['val'])
    }

    data_loaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=2),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2),
    }

    # check some stats
    train_set_size = len(image_datasets['train'])
    val_set_size = len(image_datasets['val'])
    test_set_size = len(image_datasets['test'])
    print(f'Train set size: {train_set_size}, validation set size: {val_set_size}, test set size: {test_set_size}')
    classes = image_datasets['train'].classes
    print('Train set classes: ')
    print(classes)

    return data_loaders 

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net()

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    data_loaders = create_data_loaders(args.data_dir, int(args.batch_size))

    model, best_model_params_file_name = train(model, data_loaders['train'], data_loaders['val'], loss_criterion, optimizer, int(args.epochs))

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, data_loaders['test'], loss_criterion)

    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, best_model_params_file_name))
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pt'))
    print(f"saved best model params '{args.model_dir}/{best_model_params_file_name}'")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--model_dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--output_dir", type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()

    print('hpo.py was called with these parameters: ')
    print(args)

    main(args)