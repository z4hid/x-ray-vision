import os
import sys
import torch
from tqdm import tqdm
from src.constants import DEVICE
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts
from huggingface_hub import hf_hub_download

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifacts: ModelTrainerArtifacts, data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        
    def get_best_model_from_huggingface(self):
        try:
            logging.info("Entered the get_best_model_from_huggingface method of model evaluation class")
            model_path = hf_hub_download(repo_id=self.model_evaluation_config.MODEL_REPO_NAME,
                                         filename=self.model_evaluation_config.MODEl_NAME, 
                                         local_dir=self.model_evaluation_config.BEST_MODEL_DIR)
            
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR,
                                           self.model_evaluation_config.MODEl_NAME)
            logging.info("Exited the get_best_model_from_huggingface method of model evaluation class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys)
        
    def evaluate(self, model, criterion, test_dataloader):
        try:
            logging.info("Entered the evaluate method of model evaluation class")
            total_test_loss = 0
            model.eval()
            with tqdm(test_dataloader, unit="batch", leave=False) as pbar:
                pbar.set_description(f"Testing")
                for images, labels in pbar:
                    images = images.to(DEVICE)
                    labels = labels.float().to(DEVICE)
                    output = model(images) #.view(-1)
                    loss = criterion(output, labels)
                    total_test_loss += loss.item()
            
            test_loss = total_test_loss/len(self.data_transformation_artifacts.test_transformed_object)
            print(f"Test Loss: {test_loss:.4f}")
            logging.info("Exited the evaluate method of model evaluation class")
            return test_loss

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_evaluation(self) -> ModelTrainerArtifacts:
        logging.info("Entered the initiate_model_evaluation method of model evaluation class")
        try:
            logging.info("Loading validation data for model evaluation")
            test_dataset = load_object(file_path=self.data_transformation_artifacts.test_transformed_object)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=False,
                                     batch_size=self.model_evaluation_config.BATCH_SIZE,
                                     num_workers=self.model_evaluation_config.NUM_WORKERS)
            criterion = torch.nn.BCEWithLogitsLoss()
            logging.info("Loading currently trained model")
            model = torch.load(self.model_trainer_artifacts.trained_model_path, map_location=DEVICE, weights_only=False)
            model.eval()
            trained_model_loss = self.evaluate(model=model, criterion=criterion, test_dataloader=test_loader)
            logging.info("Fetch the best model from huggingface")
            best_model_path = self.get_best_model_from_huggingface()
            logging.info("Check if the best model is same as the trained model")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("The best model is False and currently trained model is acccepted as True")
            else:
                logging.info("Load best model from huggingface")
                model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
                model.eval()
                best_model_loss = self.evaluate(model=model, criterion=criterion, test_dataloader=test_loader)
                
                logging.info("Comparing loss between best model loss and trained model loss")
                
                if best_model_loss > trained_model_loss:
                    is_model_accepted = True
                    logging.info("Trained model not accepted!!!")
                
                else:
                    is_model_accepted = False
                    logging.info("Trained model accepted!")
    
            model_trainer_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Exited the initiate_model_evaluation method of model evaluation class")
            return model_trainer_artifacts
            
            
        except Exception as e:
            raise CustomException(e, sys)