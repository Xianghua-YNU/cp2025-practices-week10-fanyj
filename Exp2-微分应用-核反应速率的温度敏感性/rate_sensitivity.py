import numpy as np
import matplotlib.pyplot as plt

def q3a(T):
    """
    计算 3-alpha 反应速率中与温度相关的部分 q / (rho^2 Y^3)
    输入: T - 温度 (K)
    返回: 速率因子 (erg * cm^6 / (g^3 * s))
    """
    # TODO: 在此实现3-α反应速率计算
    # 提示：
    # 将温度转换为以 10^8 K 为单位
    T_8 = T / 1e8
    # 处理温度为零的特殊情况
    if T_8 == 0:
        return 0
    # 使用公式：q_{3α} = 5.09×10^11 ρ^2 Y^3 T_8^(-3) exp(-44.027/T_8)
    return 5.09e11 * T_8**(-3) * np.exp(-44.027 / T_8)


def plot_rate(filename="rate_vs_temp.png"):
    """绘制速率因子随温度变化的 log-log 图"""
    # 使用 np.logspace 生成温度数据点
    T = np.logspace(7, 10, 100)
    # 计算对应的速率值
    rates = q3a(T)
    # 使用 plt.loglog 绘制双对数图
    plt.loglog(T, rates)
    # 添加适当的标签和标题
    plt.xlabel('Temperature (K)')
    plt.ylabel('Rate factor (erg * cm^6 / (g^3 * s))')
    plt.title('3 - alpha reaction rate factor vs Temperature')
    plt.grid(True)
    # 保存图片
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    # 计算并打印 nu 值
    print("   温度 T (K)    :   ν (敏感性指数)")
    print("--------------------------------------")

    temperatures_K = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    h = 1.0e-8  # 扰动因子

    for T in temperatures_K:
        q = q3a(T)
        q_perturbed = q3a(T * (1 + h))
        # 处理 q = 0 的特殊情况
        if q == 0:
            nu = 0
        else:
            # 使用前向差分计算导数
            dq_dT = (q_perturbed - q) / (T * h)
            # 计算敏感性指数 nu
            nu = (T / q) * dq_dT
        print(f"{T:15.2e} : {nu:15.2e}")

    # 调用绘图函数展示结果
    plot_rate()
