import time

from tqdm import tqdm


def process():
    print('preparing outputs in...')

    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    #while True:
    if True:
        bar = tqdm(range(200), desc='Embedding', leave=False)

        for _ in bar:
            time.sleep(0.1)


if __name__ == '__main__':
    process()

    print('ending')
