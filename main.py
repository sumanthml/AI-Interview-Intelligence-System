from src.data.data_loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.data.split_dataset import DatasetSplitter
from src.visualization.eda import ExploratoryDataAnalysis


def main() -> None:

    print("=" * 70)
    print(" AMAZON REVIEW INTELLIGENCE PLATFORM ")
    print("=" * 70)

    loader = DataLoader()

    dataframe = loader.load_data()

    preprocessor = DataPreprocessor(dataframe)

    preprocessor.preprocess()

    preprocessor.save()

    splitter = DatasetSplitter()

    splitter.split()

    eda = ExploratoryDataAnalysis()

    eda.run()

    print("=" * 70)
    print(" DATA ENGINEERING + EDA COMPLETED ")
    print("=" * 70)


if __name__ == "__main__":
    main()