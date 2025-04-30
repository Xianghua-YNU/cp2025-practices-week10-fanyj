import numpy as np
import matplotlib.pyplot as plt
from sympy import tanh, symbols, diff, lambdify


def f(x):
    """计算函数值 f(x) = 1 + 0.5*tanh(2x)

    参数：
        x: 标量或numpy数组，输入值

    返回：
        标量或numpy数组，函数值
    """
    return 1 + 0.5 * np.tanh(2 * x)


def get_analytical_derivative():
    """使用sympy获取解析导数函数

    返回：
        可调用函数，用于计算导数值
    """
    x = symbols('x')
    expr = 1 + 0.5 * tanh(2 * x)
    derivative_expr = diff(expr, x)
    return lambdify(x, derivative_expr)


def calculate_central_difference(x, f):
    """使用中心差分法计算数值导数

    参数：
        x: numpy数组，要计算导数的点
        f: 可调用函数，要求导的函数

    返回：
        numpy数组，x[1:-1]处的导数值
    """
    h = x[1] - x[0]
    return (f(x[2:]) - f(x[:-2])) / (2 * h)


def richardson_derivative_all_orders(x, f, h, max_order=3):
    """使用Richardson外推法计算不同阶数的导数值

    参数：
        x: 标量，要计算导数的点
        f: 可调用函数，要求导的函数
        h: 浮点数，初始步长
        max_order: 整数，最大外推阶数

    返回：
        列表，不同阶数计算的导数值
    """
    D = np.zeros((max_order, max_order))
    for i in range(max_order):
        D[i, 0] = (f(x + h / (2**i)) - f(x - h / (2**i))) / (2 * (h / (2**i)))
    for j in range(1, max_order):
        for i in range(max_order - j):
            D[i, j] = (4**j * D[i, j - 1] - D[i + 1, j - 1]) / (4**j - 1)
    return [D[0, i] for i in range(max_order)]


def create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical):
    """创建对比图，展示导数计算结果和误差分析

    参数：
        x: numpy数组，所有x坐标点
        x_central: numpy数组，中心差分法使用的x坐标点
        dy_central: numpy数组，中心差分法计算的导数值
        dy_richardson: numpy数组，Richardson方法计算的导数值
        df_analytical: 可调用函数，解析导数函数
    """
    # 创建四个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    # 1. 导数对比图
    ax1.plot(x, df_analytical(x), label='Analytical')
    ax1.plot(x_central, dy_central, label='Central Difference')
    ax1.plot(x[0], dy_richardson, 'o', label='Richardson')
    ax1.set_xlabel('x')
    ax1.set_ylabel('df/dx')
    ax1.set_title('Derivative Comparison')
    ax1.legend()

    # 2. 误差分析图（对数坐标）
    analytical_values = df_analytical(x_central)
    central_error = np.abs(analytical_values - dy_central)
    ax2.loglog(x_central, central_error, label='Central Difference Error')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Error (log scale)')
    ax2.set_title('Error Analysis (Log Scale)')
    ax2.legend()

    # 3. Richardson外推不同阶数误差对比图（对数坐标）
    h_values = np.logspace(-3, 0, 20)
    order_errors = []
    for h in h_values:
        richardson_values = richardson_derivative_all_orders(x[0], f, h)
        analytical_value = df_analytical(x[0])
        errors = [np.abs(analytical_value - val) for val in richardson_values]
        order_errors.append(errors)
    order_errors = np.array(order_errors)
    for i in range(order_errors.shape[1]):
        ax3.loglog(h_values, order_errors[:, i], label=f'Order {i}')
    ax3.set_xlabel('Step size h (log scale)')
    ax3.set_ylabel('Error (log scale)')
    ax3.set_title('Richardson Extrapolation Error Comparison (Log Scale)')
    ax3.legend()

    # 4. 步长敏感性分析图（双对数坐标）
    h_values = np.logspace(-6, 0, 50)
    central_errors = []
    for h in h_values:
        x_step = np.linspace(x[0] - 2 * h, x[0] + 2 * h, 5)
        dy_step = calculate_central_difference(x_step, f)
        analytical_value = df_analytical(x_step[2])
        central_errors.append(np.abs(analytical_value - dy_step[0]))
    ax4.loglog(h_values, central_errors)
    ax4.set_xlabel('Step size h (log scale)')
    ax4.set_ylabel('Central Difference Error (log scale)')
    ax4.set_title('Step Size Sensitivity Analysis (Log-Log Scale)')

    plt.tight_layout()
    plt.show()


def main():
    """运行数值微分实验的主函数"""
    # 设置实验参数
    x = np.linspace(-2, 2, 100)
    h = 0.01

    # 获取解析导数函数
    df_analytical = get_analytical_derivative()

    # 计算中心差分导数
    x_central = x[1:-1]
    dy_central = calculate_central_difference(x, f)

    # 计算Richardson外推导数
    dy_richardson = richardson_derivative_all_orders(x[0], f, h)

    # 绘制结果对比图
    create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical)


if __name__ == '__main__':
    main()
