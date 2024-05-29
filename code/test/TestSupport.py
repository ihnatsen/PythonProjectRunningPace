import unittest

from scripts.Support.path import *
from scripts.Support.data_transformation import *
from scripts.Support.format_txt import *
from scripts.Support.combinatorics import *


class TestSupp(unittest.TestCase):
    def test_get_path_df(self):
        # expected value
        path_to_mi_fitness = r'D:\PythonProjectRunningPace\code\dataset\mi_fitness\all_data.csv'
        # case: checking function with arg as list
        self.assertEqual(get_path_df(['mi_fitness'], 'all_data.csv'), path_to_mi_fitness)
        # checking function with arg as str
        self.assertEqual(get_path_df('mi_fitness', 'all_data.csv'), path_to_mi_fitness)

        # case: expected value
        path_to_google_date = r'D:\PythonProjectRunningPace\code\dataset\google_fit\training\2023-11-17T13_32_42.345+01_00_PT11M17.127S_Ходьба.tcx'
        # case: checking function with arg as list
        self.assertEqual(get_path_df(['google_fit', 'training'],
                                     '2023-11-17T13_32_42.345+01_00_PT11M17.127S_Ходьба.tcx'),
                         path_to_google_date)

        # case: new file with some directories
        path = r'D:\PythonProjectRunningPace\code\dataset\google_fit\all_data\new_data.txt'
        directories = ['dataset', 'google_fit', 'all_data']
        self.assertEqual(get_path_to_new_file(directories, 'new_data.txt'), path)

        # case: new file in modul
        path = r'D:\PythonProjectRunningPace\code\output\new_data.txt'
        self.assertEqual(get_path_to_new_file('output', 'new_data.txt'), path)

    def test_json_string_to_dict(self):
        # value for test
        value = json_string_to_dict('{"Python": "First", "Java": "4Ever", "JS": "... is a jock???"}')

        # cases: checking convert json_str -> dict
        self.assertIsInstance(value, dict)
        self.assertEqual(value['Python'], 'First')
        self.assertEqual(value['Java'], '4Ever')
        self.assertEqual(value['JS'], '... is a jock???')

    def test_format_txt(self):

        # rgb red - 204, 50, 50
        # case: The second parameter is name color, which in module "format_txt":
        self.assertEqual(f'\033[38;2;{204};{50};{50}mPython\u001b[0m',
                         paint('Python', 'red'))

        # rgb maroon = 128, 0, 0
        # case: The second parameter is rgb,
        self.assertEqual(f'\033[38;2;{128};{0};{0}mPython\u001b[0m',
                         paint('Python', [128, 0, 0]))

        # case: The second parameter is name color, which module "format_txt" don't contain:
        self.assertEqual(paint('Python', 'maroon'), 'Python')

        # case: checking that foo painting only her txt
        self.assertEqual(f'\033[38;2;{204};{50};{50}mPython\u001b[0m' + ' Java',
                         paint('Python', 'red') + ' Java')

    def test_get_all_permutations(self):
        self.assertEqual(get_all_combinations([]), [])
        self.assertEqual(get_all_combinations(['1']), [('1',)])
        self.assertEqual(get_all_combinations(['1', '2']), [('1',), ('2',) ,('1', '2'),])
        self.assertEqual(get_all_combinations(['1', '2', '3']),
                         [('1',), ('2',), ('3',),
                                 ('1', '2',), ('1', '3',), ('2', '3',),
                                 ('1', '2', '3',)])



if __name__ == '__main__':
    unittest.main()
