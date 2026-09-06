import numpy as np
import pandas as pd
from typing import List, Union

def detect_categorical_features(
    X: Union[np.ndarray, pd.DataFrame],
    threshold: int = 10,
    include_bool: bool = True
) -> List[int]:
    """
    Rileva gli indici posizionali delle colonne categoriche.

    - Se X è DataFrame, usa i dtype 'object', 'category' e (opzionalmente) 'bool'.
    - Se X è array NumPy:
        - Se dtype è object o str, la considera categorica.
        - Altrimenti, per colonne numeriche, usa il numero di valori unici.

    Args:
        X (np.ndarray o pd.DataFrame): Dati di input.
        threshold (int): Soglia per il numero di valori unici (fallback per colonne numeriche).
        include_bool (bool): Se True, le colonne bool in pandas vengono considerate categoriche.

    Returns:
        List[int]: Indici posizionali delle colonne categoriche.
    """
    if isinstance(X, pd.DataFrame):
        # Criteri per pandas: object, category, e opzionalmente bool
        def is_cat(dtype):
            # Usa isinstance per il check categorico 
            if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                return True
            if include_bool and pd.api.types.is_bool_dtype(dtype):
                return True
            return False

        categorical_mask = X.dtypes.apply(is_cat)
        return [i for i, is_cat in enumerate(categorical_mask) if is_cat]

    # Caso array NumPy (o array-like convertibile)
    X = np.asarray(X)
    categorical = []

    for i in range(X.shape[1]):
        col = X[:, i]
        # Se la colonna ha dtype object o stringa, la considera categorica
        if np.issubdtype(col.dtype, np.object_) or np.issubdtype(col.dtype, np.str_):
            categorical.append(i)
        else:
            # Per colonne numeriche, usa la soglia
            uniq = np.unique(col)
            if len(uniq) <= threshold:
                categorical.append(i)

    return categorical