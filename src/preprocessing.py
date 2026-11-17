
"""Shared preprocessing for the UNSW-NB15 project."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


CAT_COLS = ["proto", "service", "state"]

EXCLUDED_COLS = ["id", "attack_cat", "label"]


def create_preprocessor(numerical_columns):
    """Create the shared numerical and categorical preprocessor."""

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, CAT_COLS)
        ]
    )

    return preprocessor
