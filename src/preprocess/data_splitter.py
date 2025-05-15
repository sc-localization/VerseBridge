import random
from typing import Tuple

from src.utils import AppLogger
from src.type_defs import JSONDataListType, ArgLoggerType, LoggerType


class DataSplitter:
    def __init__(self, train_split_ratio: float, logger: ArgLoggerType | None = None) -> None:
        """
        Initializes DataSplitter with a data split ratio.

        Args:
            train_split_ratio (float): The proportion of data for the training split (0.0 - 1.0).
            logger (ArgLoggerType | None): A logger for logging operations (defaults to an AppLogger).

        Raises:
            ValueError: If train_split_ratio is not between 0.0 and 1.0.
        """
        if not 0.0 <= train_split_ratio <= 1.0:
            raise ValueError("train_split_ratio must be between 0.0 and 1.0")

        self.logger: LoggerType = logger or AppLogger("data_splitter").get_logger
        self.train_split_ratio: float = train_split_ratio

    def split_data(
        self, data: JSONDataListType
    ) -> Tuple[JSONDataListType, JSONDataListType]:
        """
        Shuffles the data and splits it into training and validation sets.

        Args:
            data (JSONDataListType): The list of dictionaries containing the data.

        Returns:
            Tuple[JSONDataListType, JSONDataListType]: A tuple containing two lists: the training set and the validation set.
        """
        self.logger.info(
            f"Splitting {len(data)} data entries with train ratio {self.train_split_ratio}"
        )

        if not data:
            self.logger.error("Data is empty, cannot split")
            raise ValueError("Data is empty")

        try:
            shuffled_data: JSONDataListType = data.copy()
            random.shuffle(shuffled_data)

            train_size: int = int(self.train_split_ratio * len(shuffled_data))
            train_data: JSONDataListType = shuffled_data[:train_size]
            test_data: JSONDataListType = shuffled_data[train_size:]

            self.logger.debug(f"Created train split with {len(train_data)} entries")
            self.logger.debug(f"Created test split with {len(test_data)} entries")

            return train_data, test_data
        except Exception as e:
            self.logger.error(f"Failed to split data: {str(e)}")
            raise
