__author__ = 'ptoth'


def solution(A):
    # write your code in Python 2.7
    all_sum = sum(A)
    cur_sum = 0
    for idx, el in enumerate(A):
        if cur_sum == all_sum - cur_sum - el:
            return idx
        cur_sum += el
    return -1


if __name__ == '__main__':
    arr = [-1, 3, -4, 5, 1, -6, 2, 1]
    print solution(arr)
