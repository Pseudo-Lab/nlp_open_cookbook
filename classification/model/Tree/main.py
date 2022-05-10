import logging

from src.utils import init_logger
from src.dataloader import DataLoader
from src.trainer import Trainer

logger = logging.getLogger(__name__)

def parse_args():
    pass

def main():
    init_logger()
    logger.info('Start process')
    logger.info('Data pre-processing')
    dl = DataLoader(dataset='nsmc', task='binary')
    X_train, y_train, X_test, y_test = dl.load_and_preprocess()

    logger.info("Model training")
    trainer = Trainer(model='RandomForest')
    trainer.train(X_train, y_train)

    accuarcy = trainer.evaluate(X_test, y_test)
    logger.info(f"Model accuarcy: {accuarcy}")
    
    trainer.save_model()

    return 0

if __name__ == '__main__':
    main()

