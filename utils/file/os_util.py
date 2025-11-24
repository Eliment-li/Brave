
import platform

def is_windows():
    """检查当前操作系统是否为 Windows"""
    return platform.system() == "Windows"

# system = platform.system()
# if system == "Windows":
#     print("当前操作系统是 Windows")
# elif system == "Linux":
#     print("当前操作系统是 Linux")
# else:
#     print("其他操作系统:", system)