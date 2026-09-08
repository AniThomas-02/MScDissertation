from core import run_experiment, CNN1D_Large

if __name__ == "__main__":
    run_experiment(CNN1D_Large, use_augment=True, name="Larger capacity, with augmentation")
