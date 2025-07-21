# -*- coding: utf-8 -*-
"""
此脚本用于演示如何使用 openxlab SDK 下载数据集。

TOTAL : 500G

使用步骤:
1. 安装 SDK: pip install openxlab
2. 将下面的 YOUR_ACCESS_KEY 和 YOUR_SECRET_KEY 替换为您自己的凭证。
3. 按需修改 YOUR_DOWNLOAD_PATH 变量来指定本地下载路径。
4. 运行此脚本。
"""

import os
import openxlab
from openxlab.dataset import get, info

# --- 用户配置 ---
# 重要：请将这里的占位符替换为您的真实凭证和期望的下载路径
YOUR_ACCESS_KEY = "4ep8anja7x4yoprrjkzq"  # 替换为您的 Access Key
YOUR_SECRET_KEY = "bln589rarg4vxzeq22r84epwyqqpyjd1pblwmano"  # 替换为您的 Secret Key

# 设置您想将数据集下载到的本地文件夹路径
YOUR_DOWNLOAD_PATH = "./data/ForgeryNet"

# 设置目标数据集的仓库地址
DATASET_REPO = 'OpenDataLab/ForgeryNet'


def main():
    """
    主函数，处理登录和下载流程。
    """
    # 1. 登录到 OpenXLab
    # 使用您在上文配置的 AK/SK 进行登录
    print("正在尝试登录 OpenXLab...")
    try:
        openxlab.login(ak=YOUR_ACCESS_KEY, sk=YOUR_SECRET_KEY)
        print("✅ 登录成功！")
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        print("请检查您的 Access Key 和 Secret Key 是否正确。")
        return

    # 确保本地下载目录存在，如果不存在则创建
    os.makedirs(YOUR_DOWNLOAD_PATH, exist_ok=True)

    # 2. (可选) 获取并显示数据集信息
    try:
        print(f"\n正在获取数据集 '{DATASET_REPO}' 的信息...")
        # info 函数会直接打印出数据集的元数据信息
        info(dataset_repo=DATASET_REPO)
    except Exception as e:
        print(f"❌ 获取数据集信息失败: {e}")

    # 3. 下载整个数据集
    # get 函数会将指定的数据集下载到 target_path
    print(f"\n准备开始下载整个数据集到: {YOUR_DOWNLOAD_PATH}")
    try:
        get(dataset_repo=DATASET_REPO, target_path=YOUR_DOWNLOAD_PATH)
        print(f"\n✅ 数据集已成功下载到 '{YOUR_DOWNLOAD_PATH}'！")
    except Exception as e:
        print(f"❌ 数据集下载过程中发生错误: {e}")


if __name__ == "__main__":
    main()