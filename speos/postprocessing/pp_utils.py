import pandas as pd
import numpy as np
from typing import Iterable


class PostProcessingTable:
    def __init__(self, index: Iterable = None) -> None:
        if index is not None:
            self.init_table(index=index)

    def init_table(self, index: Iterable) -> None:
        """ 
            Creates a pandas dataframe with no columns and the specified index to store later results.
        """

        self.table = pd.DataFrame(index=sorted(index))

    def add_column(self, column_header: str) -> None:
        """
            Adds a column of nans with the specified `column_header` .
        """
        if not isinstance(column_header, str):
            raise TypeError("column_header must be of type string")

        self.table[column_header] = pd.Series(np.nan, index=self.table.index)

    def add_values(self, column_header: str, index: Iterable, values, remaining=None) -> None:
        """
           Overwrites the values in the column `column_header` using `values` in the order of `index`.
           if `values` is a scalar and not an iterable, it will be converted to an iterable of length len(index).
        """
        if column_header not in self.table.columns:
            raise ValueError("Column {} specified by argument column_header does not exist yet, only {}." + 
                             " Use the add() method to add values new columns.".format(column_header, self.table.columns))

        try:
            _ = iter(values)
            if isinstance(index, set):
                raise ValueError("Sets of indices are of arbitrary order, passing iterators als values alongside will lead to invalid results.")
        except TypeError:
            values = [values] * len(index)

        self.table.loc[index, column_header] = values
        if remaining is not None:
            try:
                _ = iter(remaining)
            except TypeError:
                remaining = [remaining] * len(self.table.index.difference(index))
            self.table.loc[self.table.index.difference(index), column_header] = remaining

    def add(self, column_header: str, index: Iterable, values, remaining=None) -> None:
        """
            Adds `values` to the column `column_header` in the order of `index`. If the column does not yet exist, it is created.
            if `values` is a scalar and not an iterable, it will be converted to an iterable of length len(index).
        """

        if column_header not in self.table.columns:
            self.add_column(column_header=column_header)

        self.add_values(column_header=column_header, index=index, values=values, remaining=remaining)
            
    def get_table(self):
        return self.table