from src.transformer.transformer import TSTransformerEncoderClassiregressor
from src.trainer.trainer import Trainer
from src.utils import get_data
from src.library import *


def run_all():
    dh = get_data(train_path='data/MotionSenseHAR/MotionSenseHAR_TRAIN.ts',
                  test_path='data/MotionSenseHAR/MotionSenseHAR_TEST.ts')

    for lr in [1e-4]:
        model = TSTransformerEncoderClassiregressor(
            feat_dim=12,
            d_model=64,
            max_len=1000,
            n_heads=8,
            num_layers=6,
            dim_feedforward=512,
            num_classes=6,
            dropout=0.1,
            pos_encoding="learnable",
            activation="gelu",
            norm="BatchNorm",
            freeze=False,
        )
        trainer = Trainer(dh=dh, epochs=1)
        dh.create_dataset()
        dh.split_data(train_split=0.8)
        dataloader_train = dh.create_dataloader(dh.train_data, batch_size=8)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        trainer.fit(dataloader=dataloader_train, model=model, optimiser=optimiser)
        dataloader_test = dh.create_dataloader(dh.test_data, batch_size=8)
        accuracy = trainer.evaluate(dataloader=dataloader_test, model=model)
        print(accuracy)


if __name__ == "__main__":
    run_all()
