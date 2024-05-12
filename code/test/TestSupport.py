import unittest
from scripts.Support.path import get_path_df
from scripts.Support.data_transformation import Value, json_string_to_dict
from scripts.Support.format_txt import paint


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

    def test_value_clss(self):
        # value for test
        value_for_checking = [{'key_one': 11, 'key_two': 12, 'key_three': 13, 'key_four': 14},
                              {'key_one': 21, 'key_two': 22, 'key_three': 23, 'key_four': 24},
                              {'key_one': 31, 'key_two': 32, 'key_three': 33, 'key_four': 34}]
        # create instance for check
        value = Value(value_for_checking)

        #
        self.assertEqual(value['key_one'], [11, 21, 31])

    def test_json_string_to_dict(self):
        # value for test
        value = json_string_to_dict('{"Python": "First", "Java": "4Ever", "JS": "... is a jock???"}')

        # cases: checking convert json_str -> dict
        self.assertIsInstance(value, dict)
        self.assertEqual(value['Python'], 'First')
        self.assertEqual(value['Java'], '4Ever')
        self.assertEqual(value['JS'], '... is a jock???')

    def test_format_txt(self):

        # rgb red - 130, 0, 0
        # case: The second parameter is name color, which in module "format_txt":
        self.assertEqual(f'\033[38;2;{153};{0};{0}mPython\u001b[0m',
                         paint('Python', 'red'))

        # rgb maroon = 130, 0, 0
        # case: The second parameter is rgb,
        self.assertEqual(f'\033[38;2;{128};{0};{0}mPython\u001b[0m',
                         paint('Python', [128, 0, 0]))

        # case: The second parameter is name color, which module "format_txt" don't contain:
        self.assertEqual(paint('Python', 'maroon'), 'Python')

        # case: checking that foo painting only her txt
        self.assertEqual(f'\033[38;2;{255};{0};{0}mPython\u001b[0m' + ' Java',
                         paint('Python', 'red') + ' Java')


if __name__ == '__main__':
    unittest.main()
