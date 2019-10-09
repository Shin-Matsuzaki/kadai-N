import torch


# 代入や要素の指定
def main():
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(t)
    print(t.dtype)

    print(t[0, 2])
    print(t[:, 1])
    t[0, 0] = 11
    print(t)

    t[1] = 22
    print(t)

    t[:, 1] = 33
    print(t)

    print(t[t < 23])


if __name__ == '__main__':
    main()
