import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import mlflow
import mlflow.pytorch
import dagshub

from src.exception import CustomException
from src.logger import logging
from src.entity.pretrained_model import get_pretrained_model 
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts
from src.utils.main_utils import load_object
from src.constants import *

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifacts: ModelTrainerArtifacts, data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        
        # Set up DAGsHub and MLflow
        dagshub.init(repo_owner='z4hid', repo_name='x-ray-vision', mlflow=True)
        
    def get_best_model_from_huggingface(self):
        try:
            logging.info("Downloading the best model from HuggingFace")
            model_path = hf_hub_download(repo_id=self.model_evaluation_config.MODEL_REPO_NAME,
                                         filename=self.model_evaluation_config.MODEL_NAME)
            logging.info(f"Model downloaded from HuggingFace: {model_path}")
            
            model = torch.load(model_path, map_location=DEVICE)
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self, model, criterion, dataloader):
        try:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                with tqdm(dataloader, desc="Evaluating") as pbar:
                    for images, labels in pbar:
                        images = images.to(DEVICE)
                        labels = labels.float().to(DEVICE)
                        
                        outputs = model(images).squeeze()
                        outputs = outputs.view(-1)
                        labels = labels.view(-1)
                        
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        predicted = (outputs > 0.5).float()
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        pbar.set_postfix({'loss': val_loss / (pbar.n + 1), 'accuracy': 100 * val_correct / val_total})
            
            val_loss /= len(dataloader)
            val_accuracy = 100 * val_correct / val_total
            
            logging.info(f"Evaluation Loss: {val_loss:.4f}, Evaluation Accuracy: {val_accuracy:.2f}%")
            return val_loss, val_accuracy
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        logging.info("Initiating model evaluation")
        try:
            test_dataset = load_object(file_path=self.data_transformation_artifacts.test_transformed_object)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=False,
                                     batch_size=self.model_evaluation_config.BATCH_SIZE,
                                     num_workers=self.model_evaluation_config.NUM_WORKERS)
            criterion = nn.BCEWithLogitsLoss()
            
            with mlflow.start_run():
                logging.info("Loading the currently trained model")
                
                model_architecture = get_pretrained_model()
                state_dict = torch.load(self.model_trainer_artifacts.trained_model_path, map_location=DEVICE)
                model_architecture.load_state_dict(state_dict)
                model_architecture.to(DEVICE)
                
                trained_model_loss, trained_model_accuracy = self.evaluate(model=model_architecture, criterion=criterion, dataloader=test_loader)
                
                mlflow.log_metric("trained_model_loss", trained_model_loss)
                mlflow.log_metric("trained_model_accuracy", trained_model_accuracy)
                
                logging.info("Fetching the best model from HuggingFace")
                best_model = self.get_best_model_from_huggingface()
                model_architecture.load_state_dict(best_model)
                best_model_loss, best_model_accuracy = self.evaluate(model=model_architecture, criterion=criterion, dataloader=test_loader)
                
                mlflow.log_metric("best_model_loss", best_model_loss)
                mlflow.log_metric("best_model_accuracy", best_model_accuracy)
                logging.info(f"Comparing losses: trained_model_loss={trained_model_loss}, best_model_loss={best_model_loss}")
                if best_model_loss < trained_model_loss:
                    is_model_accepted = False
                    mlflow.log_param("best_model", "HuggingFace")
                    logging.info("The best model from HuggingFace is better. Keeping it.")
                else:
                    is_model_accepted = True
                    mlflow.log_param("best_model", "Trained")
                    logging.info("The trained model is better. Keeping the trained model.")
                
                mlflow.pytorch.log_model(model_architecture, "model")
    
            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Model evaluation completed")
            return model_evaluation_artifacts
            
        except Exception as e:
            raise CustomException(e, sys)