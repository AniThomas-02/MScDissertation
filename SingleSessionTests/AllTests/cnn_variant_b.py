from core import run_experiment, CNN1D_Original

if __name__ == "__main__":
    run_experiment(CNN1D_Original, use_augment=True, name="Original capacity, with augmentation")
