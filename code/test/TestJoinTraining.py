import unittest
from scripts.JoinWeatherTraining import *
from datetime import timedelta


class TestJoin(unittest.TestCase):
    def test_calculate_weight(self):
        # case 15:30:00 - 16:30:00
        self.assertEqual((0.5, 0.5), calculate_weight(timedelta(hours=15, minutes=30), 60*60))  # add assertion here
        # case 09:05:00 -  09:30:25
        self.assertEqual((1, 0), calculate_weight(timedelta(hours=9, minutes=5), 25*60+25))


if __name__ == '__main__':
    unittest.main()
