"""Run the full cheminformatics pipeline: python main.py"""
import data_collection
import data_cleaning
import molecular_features
import dataset_preparation

if __name__ == "__main__":
    data_collection.main()
    data_cleaning.main()
    molecular_features.main()
    dataset_preparation.main()
    print("\nPipeline complete. ML-ready data is in data/final/")