"""从 liu-test-exp 下的 .md 文件中提取 [XX123] 形式的引用编号。

编号规则：方括号包裹，内部是两个字母后接至少一位数字，例如 [PR024]、[RE326]。
用法：
    python extract_codes.py                     # 递归处理脚本所在目录下的所有 .md
    python extract_codes.py 某个文件.md         # 只处理指定文件
    python extract_codes.py 某个文件夹          # 只处理指定文件夹下的 .md（递归）
"""
import re
import sys
from pathlib import Path

# 两个字母 + 一位及以上数字
CODE_RE = re.compile(r"\[([A-Za-z]{2}[0-9]+)\]")

# 让中文输出在 Windows 控制台不乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def extract_from_text(text: str):
    """按出现顺序返回文本中所有编号（不去重）。"""
    return CODE_RE.findall(text)


def iter_md_files(paths):
    """把传入的路径展开为 .md 文件列表。无参数时默认使用脚本所在目录。"""
    if not paths:
        paths = [Path(__file__).resolve().parent]
    files = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() == ".md":
            files.append(p)
        elif p.is_dir():
            files.extend(f for f in p.rglob("*.md") if f.is_file())
    # 按路径排序，保证输出顺序稳定
    return sorted(set(files), key=lambda f: str(f).lower())


def main():
    files = iter_md_files(sys.argv[1:])
    if not files:
        print("未找到任何 .md 文件。")
        return

    all_codes = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        codes = extract_from_text(text)
        all_codes.extend(codes)
        uniq = sorted(set(codes))
        print(f"== {f} ==")
        print(f"   匹配总数: {len(codes)}，去重后: {len(uniq)}")
        print("   编号列表:", " ".join(uniq))
        print()

    if len(files) > 1:
        uniq_all = sorted(set(all_codes))
        print(f"== 全部文件汇总 ==")
        print(f"   总数(含重复): {len(all_codes)}，去重后: {len(uniq_all)}")
        print("   编号列表:", " ".join(uniq_all))


if __name__ == "__main__":
    main()
