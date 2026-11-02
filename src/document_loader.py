import os

import pandas as pd


class LoadDocuments:
    """
    Class for loading and validating FAQ data from a CSV file.

    This is the first step of the pipeline: the raw FAQ knowledge base is read
    from disk here and handed to the indexer (src/indexer.py) for embedding.
    """

    def load_faq_data(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file and return a DataFrame with 'id', 'title', and 'content' columns.

        Args:
            file_path (str): Path to the input CSV file.

        Returns:
            pd.DataFrame: Loaded data with required columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a CSV, missing columns, or is empty.
        """
        # --- Validate the path before touching the filesystem ---
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # A directory "exists" too, so it needs its own explicit check
        if os.path.isdir(file_path):
            raise FileNotFoundError(f"Expected a file, but found a directory: {file_path}")

        if not file_path.lower().endswith(".csv"):
            raise ValueError(f"File '{file_path}' is not a CSV file.")

        # --- Read the CSV, translating pandas errors into clearer ValueErrors ---
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError as err:
            raise ValueError("CSV file is empty.") from err
        except Exception as err:
            raise ValueError(f"Failed to read CSV file: {err}") from err

        # File parsed fine but has a header only / no data rows
        if df.empty or len(df) == 0:
            raise ValueError("CSV file is empty or contains no records.")

        # --- Ensure the schema the rest of the pipeline relies on is present ---
        required_columns = {"id", "title", "content"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        # Keep only the needed columns (drops any extras); copy avoids pandas
        # SettingWithCopy warnings if callers modify the result later
        return df[["id", "title", "content"]].copy()
