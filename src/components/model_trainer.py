import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.exception import CustomException
from src.logger import logging

from src.entity.pretrained_model import get_pretrained_model 
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts
from src.utils.main_utils import load_object
from src.constants import *


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
        try:
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with tqdm(train_dataloader, desc="Training") as pbar:
                for images, labels in pbar:
                    images = images.to(DEVICE)
                    labels = labels.float().to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(images).squeeze()
                    
                    # Ensure outputs and labels have the same shape
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)
                    
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'accuracy': 100 * train_correct / train_total})
            
            train_loss /= len(train_dataloader)
            train_accuracy = 100 * train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                with tqdm(val_dataloader, desc="Validation") as pbar:
                    for images, labels in pbar:
                        images = images.to(DEVICE)
                        labels = labels.float().to(DEVICE)
                        
                        outputs = model(images).squeeze()
                        
                        # Ensure outputs and labels have the same shape
                        outputs = outputs.view(-1)
                        labels = labels.view(-1)
                        
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        predicted = (outputs > 0.5).float()
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        pbar.set_postfix({'loss': val_loss / (pbar.n + 1), 'accuracy': 100 * val_correct / val_total})
            
            val_loss /= len(val_dataloader)
            val_accuracy = 100 * val_correct / val_total

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            return train_loss, train_accuracy, val_loss, val_accuracy

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
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
                                      shuffle=False,
                                      batch_size=self.batch_size, 
                                      num_workers=self.num_workers)
            logging.info("Loaded train and valid dataloader")
            
            # Load the model from the pretrainedmodel file
            model = get_pretrained_model()
            logging.info("Loaded pretrained ResNet34 model")
            model = model.to(DEVICE)
            
            criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with sigmoid
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            
            logging.info("Model Training started")
            
            os.makedirs(os.path.dirname(self.model_trainer_config.TRAINED_MODEL_PATH), exist_ok=True)
            best_val_loss = float('inf')
            
            for epoch in range(self.epochs):
                logging.info(f"Epoch: {epoch+1}/{self.epochs}")
                print(f"\nEpoch: {epoch+1}/{self.epochs}")
                
                train_loss, train_accuracy, val_loss, val_accuracy = self.train(model, criterion, optimizer, train_loader, valid_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), self.model_trainer_config.TRAINED_MODEL_PATH)
                    logging.info(f"Saved best model to {self.model_trainer_config.TRAINED_MODEL_PATH}")
            
            logging.info("Model Training completed")
            
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH
            )
            logging.info(f"Model trainer artifacts: {model_trainer_artifacts}")
            
            logging.info("Exited the initiate_model_trainer method of model trainer class")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys)
