import os

csv_file = "test_predictions.csv"  # 请替换为实际文件路径

# 检查原文件是否以换行结尾，如果不是，先追加换行
with open(csv_file, 'ab+') as f:
    f.seek(-1, os.SEEK_END)
    last_char = f.read(1)
    if last_char != b'\n':
        f.write(b'\n')

# 追加新行
with open(csv_file, 'a') as f:
    for num in range(6420, 8000):
        f.write(f"{num},0\n")