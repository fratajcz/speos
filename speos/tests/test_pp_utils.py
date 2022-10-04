from speos.postprocessing.pp_utils import PostProcessingTable
import pandas as pd
import numpy as np
import unittest


class PPTableTest(unittest.TestCase):

    def setUp(self):
        index = ["a", "b", "d", "c"]
        self.table = PostProcessingTable(index)

    def test_init(self):
        self.assertTrue((self.table.table.index == sorted(["a", "b", "d", "c"])).all())

    def test_get_table(self):
        self.assertTrue(isinstance(self.table.get_table(), pd.DataFrame))

    def test_add_column(self):
        column_header = "first"
        self.table.add_column(column_header=column_header)
        inserted_column = self.table.get_table()[column_header].tolist()

        # since we did not add values, the inserted columns should only contain np.nans
        self.assertEqual(len(inserted_column), 4)
        self.assertEqual(len(np.asarray(inserted_column)[~np.isnan(inserted_column)]), 0)

    def test_add(self):
        column_header = "first"
        index = ["d", "a"]
        values = [14, 7]
        final_column = [7, np.nan, np.nan, 14]
        self.table.add(column_header=column_header, index=index, values=values)
        inserted_column = self.table.get_table()[column_header].tolist()

        # we can not compare the nans since nan == nan equals False in numpy logic, therefore we have to remove the nans
        isequal = np.equal(np.asarray(inserted_column)[~np.isnan(inserted_column)], np.asarray(final_column)[~np.isnan(final_column)])
        self.assertTrue(isequal.all())

        index = ["c", "b"]
        values = [5, 18]
        final_column = [7, 18, 5, 14]
        self.table.add(column_header=column_header, index=index, values=values)
        inserted_column = self.table.get_table()[column_header].tolist()
        # we can not compare the nans since nan == nan equals False in numpy logic, therefore we have to remove the nans
        isequal = np.equal(np.asarray(inserted_column)[~np.isnan(inserted_column)], np.asarray(final_column)[~np.isnan(final_column)])
        self.assertTrue(isequal.all())

    def test_add_with_remaining(self):
        column_header = "first"
        index = ["d", "a"]
        values = [14, 7]
        remaining = 0
        final_column = [7, 0, 0, 14]
        self.table.add(column_header=column_header, index=index, values=values, remaining=0)
        inserted_column = self.table.get_table()[column_header].tolist()

        # we can not compare the nans since nan == nan equals False in numpy logic, therefore we have to remove the nans
        isequal = np.equal(np.asarray(inserted_column)[~np.isnan(inserted_column)], np.asarray(final_column)[~np.isnan(final_column)])
        self.assertTrue(isequal.all())

    def test_add_set(self):
        column_header = "first"
        index = {"d", "a"}
        values = 1
        remaining = 0
        final_column = [1, 0, 0, 1]        
        self.table.add(column_header=column_header, index=index, values=values, remaining=remaining)
        inserted_column = self.table.get_table()[column_header].tolist()

        # we can not compare the nans since nan == nan equals False in numpy logic, therefore we have to remove the nans
        isequal = np.equal(np.asarray(inserted_column)[~np.isnan(inserted_column)], np.asarray(final_column)[~np.isnan(final_column)])
        self.assertTrue(isequal.all())

        self.assertRaises(TypeError, self.table.add, column_header=column_header, index=index, values=[0, 1])

    def test_add_scalar(self):
        column_header = "first"
        index = ["d", "a"]
        values = 1

        final_column = [1, np.nan, np.nan, 1]
        self.table.add(column_header=column_header, index=index, values=values)
        inserted_column = self.table.get_table()[column_header].tolist()

        # we can not compare the nans since nan == nan equals False in numpy logic, therefore we have to remove the nans
        isequal = np.equal(np.asarray(inserted_column)[~np.isnan(inserted_column)], np.asarray(final_column)[~np.isnan(final_column)])
        self.assertTrue(isequal.all())

    def test_add_scalar_with_remaining(self):
        column_header = "first"
        index = ["d", "a"]
        values = 1
        remaining = 0

        final_column = [1, 0, 0, 1]
        self.table.add(column_header=column_header, index=index, values=values, remaining=remaining)
        inserted_column = self.table.get_table()[column_header].tolist()

        # we can not compare the nans since nan == nan equals False in numpy logic, therefore we have to remove the nans
        isequal = np.equal(np.asarray(inserted_column)[~np.isnan(inserted_column)], np.asarray(final_column)[~np.isnan(final_column)])
        self.assertTrue(isequal.all())