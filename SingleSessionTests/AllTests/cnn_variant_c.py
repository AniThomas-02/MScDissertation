from core import run_experiment, CNN1D_Reduced

if __name__ == "__main__":
    run_experiment(CNN1D_Reduced, use_augment=False, name="Reduced capacity, no augmentation")
