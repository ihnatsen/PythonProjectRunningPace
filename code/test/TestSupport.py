import unittest
from scripts.Support.path import get_path_df


class TestSupp(unittest.TestCase):
    def test_get_path_df(self):
        # expected value
        path = r'D:\PythonProjectRunningPace\code\dataset\mi_fitness\all_data.csv'
        # checking function with arg as list
        self.assertEqual(get_path_df(['mi_fitness'], 'all_data.csv'), path)
        # checking function with arg as str
        self.assertEqual(get_path_df('mi_fitness', 'all_data.csv'), path)
        # checking function with arg as type
        self.assertEqual(get_path_df(('mi_fitness'), 'all_data.csv'), path)


if __name__ == '__main__':
    unittest.main()
