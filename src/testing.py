import functions as fun


@fun.timeit
def time_if_test(N, A, B):
    count = 0
    for i in range(N):
        if A == B:
            count += 1

    return count


if __name__ == '__main__':
    N = 50000000

    A = 1
    B = 'L1'
    C = 'ELASTIC'

    print(time_if_test(N, A, A))

    print(time_if_test(N, B, B))

    print(time_if_test(N, C, C))

    # Very minimal difference, not worth doing anything about
