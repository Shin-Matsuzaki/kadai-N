import matplotlib.pyplot as plt
import torch
from torch import nn, optim


def main():
    torch.manual_seed(0)

    w_true = torch.tensor([1., 2., 3.])
    N = 100
    X = torch.cat([torch.ones(N, 1), torch.randn((N, 2))], dim=1)
    noise = torch.randn(N) * 0.5
    y = torch.mv(X, w_true) + noise

    # 学習
    learning_rate = 0.1
    loss_list = []
    num_epochs = 20

    # ネットワーク
    # 入力が3次元で出力が1次元のネットワーク
    net = nn.Linear(in_features=3, out_features=1, bias=False)
    # 確率的勾配降下法による最適化を選択
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # 損失関数はMSEを採用
    criterion = nn.MSELoss()

    # 重みは指定しなくても勝手に準備してくれてる
    parameters = list(net.parameters())
    print(parameters)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        y_pred = net(X)
        mse_loss = criterion(y_pred.view_as(y), y)
        print(y_pred)
        mse_loss.backward()
        loss_list.append(mse_loss.item())

        optimizer.step()

    # 損失の可視化
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':
    main()
