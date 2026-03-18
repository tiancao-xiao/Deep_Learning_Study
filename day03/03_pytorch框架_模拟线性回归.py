# 导入相关模块
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn模块中有平方损失函数和假设函数
from torch import optim  # optim模块中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集
import matplotlib.pyplot as plt  # 可视化

plt.switch_backend('TkAgg')  # 修复：切换后端解决PyCharm绘图报错
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 定义函数, 创建线性回归样本数据
def create_dataset():
    # 1. 创建数据集对象
    x, y, coef = make_regression(
        n_samples=100,  # 100条样本(100个样本点)
        n_features=1,  # 1个特征(1个特征点)
        noise=10,  # 噪声, 噪声越大, 样本点越散, 噪声越小, 样本点越集中
        coef=True,  # 是否返回系数, 默认为False, 返回值为None
        bias=14.5,  # 偏置
        random_state=24  # 随机种子, 随机种子相同, 输出数据相同
    )

    # 2. 把上述的数据, 封装成 张量对象
    # tensor:把数据转换为张量对象  参1: 数据, 参2: 数据类型, 参3: 是否需要自动微分(默认False, 需要True)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    coef = torch.tensor(coef, dtype=torch.float32)  # 转换为张量，消除NumPy警告

    # 3. 返回结果.
    return x, y, coef

# 2. 定义函数, 表示模型训练
def train(x, y, coef):
    # 1. 创建数据集对象. 把 tensor -> 数据集对象 -> 数据加载器
    # TensorDataset: 构造数据集对象  参1: 特征数据, 参2: 标签数据
    dataset = TensorDataset(x, y)
    # 2. 创建数据加载器对象
    # DataLoader: 数据加载器  参1: 数据集对象, 参2: 批次大小, 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # 3. 创建初始的 线性回归模型.
    # nn.Linear: 线性回归模型  参1: 输入特征维度, 参2: 输出特征维度
    model = nn.Linear(1, 1)
    # 4. 创建损失函数对象
    criterion = nn.MSELoss()
    # 5. 创建优化器对象.
    # optim.SGD: 随机梯度下降优化器  参1: 模型参数, 参2: 学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 6. 具体的训练过程
    # 6.1 定义变量, 分别表示: 训练轮数, 每轮的(平均)损失值, 训练总损失值, 训练的样本数
    # 这里定义了100轮的训练,
    # 每轮的平均损失值保存在loss_list列表中,
    # total_loss记录训练的总损失值,
    # total_sample记录训练的样本数
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    # 6.2 开始训练, 按轮训练.
    for epoch in range(epochs):  # epoch的值: 0, 1, 2...99
        # 6.3 每轮是分 批次 训练的, 所以从 数据加载器中 获取 批次数据.
        for train_x, train_y in dataloader:  # 7批(16, 16, 16, 16, 16, 16, 4) 16*6+4=100
            # 6.4 模型预测.
            y_pred = model(train_x)
            # 6.5 计算(每批的平均)损失值.
            # criterion: 损失函数对象
            # 参1: 预测值, 参2: 真实值(标签)  train_y的形状: (批次大小, ) -> 转成(批次大小, 1)
            loss = criterion(y_pred, train_y.reshape(-1, 1))  # -1 自动计算.
            # 6.6 计算总损失 和 样本(批次)数
            # loss.item(): 获取loss的数值, 不是张量对象了, 方便后续计算总损失值
            total_loss += loss.item()
            total_sample += 1
            # 6.7 梯度清零 + 反向传播 + 梯度更新.
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播, 计算梯度
            optimizer.step()  # 梯度更新

        # 6.8 把本轮的(平均)损失值, 添加到列表中
        loss_list.append(total_loss / total_sample)
        print(f'轮数: {epoch + 1}, 平均损失值: {total_loss / total_sample}')

    # 7. 打印(最终的)训练结果
    print(f'{epochs} 轮的平均损失分别为: {loss_list}')
    print(f'模型参数, 权重: {model.weight}, 偏置: {model.bias}')

    # 8. 绘制损失曲线
    #                 100轮         每轮的平均损失值
    plt.figure(1)  # 指定图表1
    plt.plot(range(epochs), loss_list)
    plt.title('损失值曲线变化图')
    plt.grid()      # 绘制网格线
    plt.show(block=False)  # 非阻塞模式，双图同时弹出

    # 9. 绘制预测值和真实值的关系
    # 9.1 绘制样本点分布情况
    plt.figure(2)  # 指定图表2
    # scatter: 散点图  参1: x轴数据, 参2: y轴数据
    plt.scatter(x, y)
    # 9.2 绘制训练模型的预测值
    # 直接模型预测+分离梯度，消除PyTorch警告
    y_pred = model(x).detach()
    # 9.3 计算真实值.
    # 纯张量运算，消除NumPy警告
    y_true = x * coef + 14.5
    # 9.4 绘制预测值 和 真实值的 折线图.
    plt.plot(x, y_pred, color='red', label='预测值')
    plt.plot(x, y_true, color='green', label='真实值')
    # 9.5 图例, 网格
    plt.legend()
    plt.grid()
    # 9.6 显示图像
    plt.show()


# 3. 测试.
if __name__ == '__main__':
    # 3.1 创建数据集.
    x, y, coef = create_dataset()
    # print(f'x: {x}, y: {y}, coef: {coef}')

    # 3.2 模型训练.
    train(x, y, coef)