import unittest

__author__ = 'ptoth'


class FindSeparationTests(unittest.TestCase):

    def arr1_test(self):
        arr = [1, 1, -1, -1]
        self.assertTupleEqual(self.get_separation_with_minimal_error(arr), (1, 4, 0))

    def arr2_test(self):
        arr = [-1, 1, 1, 1]
        self.assertTupleEqual(self.get_separation_with_minimal_error(arr), (0, 4, 0))

    def arr3_test(self):
        arr = [1, -1, 1, -1, -1]
        self.assertTupleEqual(self.get_separation_with_minimal_error(arr), (0, 4, 1))

    def arr4_test(self):
        arr = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
        self.assertTupleEqual(self.get_separation_with_minimal_error(arr), (2, 9, 3))

    def get_separation_with_minimal_error(self, array):

        ones = 0
        minus_ones = 0

        # determine total number of 1's and -1's
        for idx, el in enumerate(array):
            if el == 1:
                ones += 1
            else:
                minus_ones += 1

        left_ones = 0
        left_minusones = 0
        sep_error = -1
        sep_index = 0
        error = -1
        for idx, el in enumerate(array):
            # running through the array at each element wi determine
            # number of 1's and -1's in the left vs. right subarray
            if el == 1:
                left_ones += 1
            else:
                left_minusones += 1
            right_ones = ones - left_ones
            right_minus_ones = minus_ones - left_minusones

            # we want to homogenize the subarrays as much as possible in a way
            # that we determine which subarray is more pos or more neg
            # we measure the separation of the cut as
            # the maximum of
            # the number of 1's in the left plus number of -1's in right and
            # the number of -1's in the left plus the number of 1's in the right
            cur_error = max(left_ones + right_minus_ones, left_minusones + right_ones)
            if cur_error > sep_error:
                sep_error = cur_error
                sep_index = idx
                # the actual error will be the maximum of the difference in homogeneity between the sides
                error = max(min(left_minusones, left_ones), min(right_minus_ones, right_ones))

        return sep_index, sep_error, error


if __name__ == '__main__':
    unittest.main()

