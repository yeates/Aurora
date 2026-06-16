import platform
import sys

import torch


def get_cuda_version():
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        return f"cu{cuda_version.replace('.', '')[:2]}"  # 例如：cu121
    return "cpu"


def get_torch_version():
    return f"torch{torch.__version__.split('+')[0]}"[:-2]  # 例如：torch2.2


def get_python_version():
    version = sys.version_info
    return f"cp{version.major}{version.minor}"  # 例如：cp310


def get_abi_flag():
    return "abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "abiFALSE"


def get_platform():
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine == "x86_64":
        return "linux_x86_64"
    elif system == "windows" and machine == "amd64":
        return "win_amd64"
    elif system == "darwin" and machine == "x86_64":
        return "macosx_x86_64"
    else:
        raise ValueError(f"Unsupported platform: {system}_{machine}")


def generate_flash_attn_filename(flash_attn_version="2.7.3"):
    cuda_version = get_cuda_version()
    torch_version = get_torch_version()
    python_version = get_python_version()
    abi_flag = get_abi_flag()
    platform_tag = get_platform()

    filename = (
        f"flash_attn-{flash_attn_version}+{cuda_version}{torch_version}cxx11{abi_flag}-"
        f"{python_version}-{python_version}-{platform_tag}.whl"
    )
    return filename,flash_attn_version

if __name__ == "__main__":
    try:
        filename,flash_attn_version = generate_flash_attn_filename()
        print(f"Install manually:\nwget https://github.com/Dao-AILab/flash-attn/releases/download/v{flash_attn_version}/{filename}\npip install {filename}")
    except Exception as e:
        print("Error generating filename:", e)