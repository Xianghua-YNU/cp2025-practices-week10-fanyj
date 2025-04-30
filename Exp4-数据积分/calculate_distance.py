import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import os



def main():
    try:
        data_file ="Exp4-数据积分/calculate_distance.py"  # 使用相对路径，确保文件与脚本同目录

        # 2. 读取数据（假设数据有两列：时间 速度）
        t, v = np.loadtxt(data_file, unpack=True)  # 自动解包到时间列和速度列

        # 3. 计算总距离（使用梯形法数值积分）
        total_distance = np.trapezoid(v, t)  # 第一个参数是y值，第二个参数是x值
        print(f"总运行距离: {total_distance:.2f} 米")

        # 4. 计算累积距离（带初始值保证数组长度一致）
        distance = cumulative_trapezoid(v, t, initial=0)  # 添加initial=0保持长度匹配

        # 5. 绘制图表
        plt.figure(figsize=(10, 6))
        plt.plot(t, v, 'b-', label='Velocity (m/s)')
        plt.plot(t, distance, 'r--', label='Distance (m)')
        plt.title('Velocity and Distance vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s) / Distance (m)')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print("错误：找不到数据文件")
        print("请确保数据文件存在于正确路径")
    except ValueError as e:
        print("数据格式错误:", e)
        print("请确保数据文件包含两列数值（时间 速度）")
        print("各列数值用空格或逗号分隔")


if __name__ == '__main__':
    main()
