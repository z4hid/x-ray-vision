import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.exception import CustomException
from src.logger import logging
from src.entity.pretrained_model import get_pretrained_model 
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifacts, DataTransformationArtifacts
from src.utils.main_utils import load_object
from src.constants import *


class ModelEvaluator:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        Initializes the model evaluator with the config and data artifacts.
        """
        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.batch_size = self.model_evaluation_config.BATCH_SIZE
        self.num_workers = self.model_evaluation_config.NUM_WORKERS

    def evaluate(self, model, criterion, dataloader):
        """
        This method evaluates the model on the validation/test set.
        """
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
                        
                        # Ensure outputs and labels have the same shape
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
            
            print(f"Evaluation Loss: {val_loss:.4f}, Evaluation Accuracy: {val_accuracy:.2f}%")
            logging.info(f"Evaluation Loss: {val_loss:.4f}, Evaluation Accuracy: {val_accuracy:.2f}%")
            
            return val_loss, val_accuracy

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        This method initiates model evaluation by loading the model and running the evaluation.
        """
        try:
            logging.info("Entered the initiate_model_evaluation method of ModelEvaluator class")
            
            test_dataset = load_object(file_path=self.data_transformation_artifacts.test_transformed_object)
            logging.info("Loaded test dataset")
            
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=False, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers)
            logging.info("Loaded test dataloader")
            
            # Load the pretrained model
            model = get_pretrained_model()
            logging.info("Loaded pretrained ResNet34 model")
            
            # Load model weights from the best-trained model checkpoint
            if os.path.exists(self.model_evaluation_config.TRAINED_MODEL_PATH):
                model.load_state_dict(torch.load(self.model_evaluation_config.TRAINED_MODEL_PATH))
                logging.info(f"Loaded trained model from {self.model_evaluation_config.TRAINED_MODEL_PATH}")
            else:
                raise CustomException("Trained model not found.", sys)
            
            model = model.to(DEVICE)
            
            criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with sigmoid, so no need to use sigmoid in the model
            
            logging.info("Model Evaluation started")
            
            val_loss, val_accuracy = self.evaluate(model, criterion, test_loader)
            
            logging.info("Model Evaluation completed")
            
            model_evaluation_artifacts = ModelEvaluationArtifacts(
                evaluation_loss=val_loss,
                evaluation_accuracy=val_accuracy
            )
            logging.info(f"Model evaluation artifacts: {model_evaluation_artifacts}")
            
            logging.info("Exited the initiate_model_evaluation method of ModelEvaluator class")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys)
