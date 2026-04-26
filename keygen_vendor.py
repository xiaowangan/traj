# -*- coding: utf-8 -*-
"""供应商专用：根据用户机器码 + 授权天数生成激活码"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from function.license_manager import generate_activation_code

if __name__ == "__main__":
    print("=" * 56)
    print("  平面轨迹生成软件 — 激活码生成工具（供应商专用）")
    print("=" * 56)
    hwid = input("请输入用户提供的机器码：").strip()
    try:
        days = int(input("授权天数（如 365）：").strip())
        assert days > 0
    except:
        print("授权天数无效"); input("按 Enter 退出..."); sys.exit(1)
    code = generate_activation_code(hwid, days)
    print(f"\n  激活码（{days} 天）：\n  {code}")
    print("\n  请将激活码和授权天数一并告知用户。")
    input("\n按 Enter 退出...")
