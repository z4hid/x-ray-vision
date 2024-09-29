import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models

from sklearn.metrics import precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifacts
from src.utils.main_utils import load_object
from src.constants import DEVICE


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifacts: DataTransformationArtifacts):
        
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.learning_rate = self.model_trainer_config.LR
        self.epochs = self.model_trainer_config.EPOCHS
        self.batch_size = self.model_trainer_config.BATCH_SIZE
        self.num_workers = self.model_trainer_config.NUM_WORKERS
        
        
    def train(self, model, criterion, optimizer, train_dataloader, val_dataloader):
        # Initialize variables to store total loss and predictions
        total_train_loss = 0
        total_val_loss = 0
        train_preds = []
        train_labels = []
        val_preds = []
        val_labels = []
        
        # Set model to training mode
        model.train()
        with tqdm(train_dataloader, unit="batch", leave=False) as pbar:
            pbar.set_description(f"Training")
            for images, labels in pbar:
                # Move data to the specified device
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.float().to(DEVICE, non_blocking=True)
                
                # Forward pass
                output = model(images).view(-1)  # Ensure output is flattened
                
                # Compute loss
                loss = criterion(output, labels)
                total_train_loss += loss.item()
                
                # Store predictions and labels
                train_preds.extend((output > 0).float().cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Set model to evaluation mode
        model.eval()
        with torch.no_grad():  # Disable gradient calculation for validation
            with tqdm(val_dataloader, unit="batch", leave=False) as pbar:
                pbar.set_description(f"Validation")
                for images, labels in pbar:
                    # Move data to the specified device
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.float().to(DEVICE, non_blocking=True)
                    
                    # Forward pass
                    output = model(images).view(-1)  # Ensure output is flattened
                    
                    # Compute loss
                    loss = criterion(output, labels)
                    total_val_loss += loss.item()
                    
                    # Store predictions and labels
                    val_preds.extend((output > 0).float().cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
        
        # Calculate average loss and metrics
        train_loss = total_train_loss / len(train_dataloader)
        val_loss = total_val_loss / len(val_dataloader)
        
        train_acc = (torch.tensor(train_preds) == torch.tensor(train_labels)).float().mean().item()
        val_acc = (torch.tensor(val_preds) == torch.tensor(val_labels)).float().mean().item()
        
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        # print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        # print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        return train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f1
    
    
    def initiate_model_trainer(self) -> ModelTrainerAtrifacts:
        
        try:
            logging.info("Entered the initiate_model_trainer method of model trainer class")
            
            train_dataset = load_object(file_path=self.data_transformation_artifacts.train_transformed_object)
            valid_dataset = load_object(file_path=self.data_transformation_artifacts.valid_transformed_object)
            logging.info("Loaded train and valid dataset")
            
            train_loader = DataLoader(dataset=train_dataset,
                                      shuffle=True, 
                                      batch_size=self.batch_size, 
                                      num_workers=self.num_workers)
            valid_loader = DataLoader(dataset=valid_dataset,
                                      shuffle=True,
                                      batch_size=self.batch_size, 
                                      num_workers=self.num_workers)
            logging.info("Loaded train and valid dataloader")
            
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)
            model = model.to(DEVICE)
            logging.info("Loaded the model")
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            

        except Exception as e:
            raise CustomException(e, sys)