import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()


    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """
        This method starts the data ingestion process.

        Returns:
            DataIngestionArtifacts: Artifact containing the paths or references to the ingested data.

        Raises:
            CustomException: If an error occurs during the data ingestion process.
        """
        try:
            logging.info("Entereed the start_data_ingestion method of training pipeline class")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Exited the start_data_ingestion method of training pipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def start_data_transformation(self, data_ingestion_artifacts: DataIngestionArtifacts) -> DataTransformationArtifacts:
        """
        This method starts the data transformation process.

        Parameters:
            data_ingestion_artifacts (DataIngestionArtifacts): Artifact containing the paths or references to the ingested data.

        Returns:
            DataTransformationArtifacts: Artifact containing the paths or references to the transformed data.

        Raises:
            CustomException: If an error occurs during the data transformation process.
        """
        logging.info("Entered the start_data_transformation method of training pipeline class")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config
            )
            data_transformation_artifacts = (data_transformation.initiate_data_transformation())
            logging.info("Exited the start_data_transformation method of training pipeline class")
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifacts: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        try:
            logging.info("Entered the start_model_trainer method of training pipeline class")
            model_trainer = ModelTrainer(
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifacts = (model_trainer.initiate_model_trainer())
            logging.info("Exited the start_model_trainer method of training pipeline class")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def start_model_evaluation(self, model_trainer_artifacts: ModelTrainerArtifacts,
                               data_transformation_artifacts: DataTransformationArtifacts) -> ModelEvaluationArtifacts:
        try:
            logging.info("Entered the start_model_evaluation method of training pipeline class")
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_artifacts=model_trainer_artifacts,
            )
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info("Exited the start_model_evaluation method of training pipeline class")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def run_pipeline(self):
        try:
            logging.info("Entered the run_pipeline method of training pipeline class")
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifacts=data_ingestion_artifact
                )
            model_trainer_artifacts = self.start_model_trainer(
                data_transformation_artifacts=data_transformation_artifact
                )
            model_evaluation_artifacts = self.start_model_evaluation(
                model_trainer_artifacts=model_trainer_artifacts,
                data_transformation_artifacts=data_transformation_artifact
                )
            logging.info("Exited the run_pipeline method of training pipeline class")
            
        except Exception as e:
            raise CustomException(e, sys)
