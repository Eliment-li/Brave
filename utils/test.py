import csv
import os
import tempfile

def keep_top_n(csv_path: str, n: int, has_header: bool = True, encoding: str = 'utf-8') -> None:
    """
    在原地保留 CSV 文件 `csv_path` 的前 n 行。
    - has_header=True 时，保留第一行表头，再保留 n 行数据（不计表头）。
    - has_header=False 时，保留前 n 行。
    - n must be >= 0.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    dirpath = os.path.dirname(csv_path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='', encoding=encoding) as outfile, \
             open(csv_path, 'r', newline='', encoding=encoding) as infile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 写表头（如果存在）
            if has_header:
                try:
                    header = next(reader)
                except StopIteration:
                    # 原文件为空，直接替换为空文件
                    pass
                else:
                    writer.writerow(header)

            # 写前 n 行数据
            written = 0
            for row in reader:
                if written >= n:
                    break
                writer.writerow(row)
                written += 1

        # 原子替换原文件
        os.replace(tmp_path, csv_path)
    except Exception:
        # 若发生异常，移除临时文件再抛出
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
import csv
import os
import tempfile

def round_floats_in_csv(csv_path: str, decimals: int = 4, encoding: str = 'utf-8', has_header: bool = True) -> None:
    if decimals < 0:
        raise ValueError("decimals must be >= 0")

    dirpath = os.path.dirname(csv_path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='', encoding=encoding) as outfile, \
             open(csv_path, 'r', newline='', encoding=encoding) as infile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            first_row = True
            for row in reader:
                if first_row and has_header:
                    writer.writerow(row)
                    first_row = False
                    continue
                first_row = False

                new_row = []
                for cell in row:
                    s = cell.strip()
                    if s == '':
                        new_row.append(cell)
                        continue
                    try:
                        f = float(s)
                    except ValueError:
                        new_row.append(cell)
                    else:
                        # 若原文本中有小数点或科学计数法标志，视为浮点数并格式化
                        if any(ch in s for ch in ('.', 'e', 'E')):
                            new_row.append(f"{f:.{decimals}f}")
                        else:
                            new_row.append(cell)
                writer.writerow(new_row)

        os.replace(tmp_path, csv_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


if __name__ == '__main__':
    #round_floats_in_csv(r'D:\sync\brave_data\bad reweard function\episode_return.csv')
    keep_top_n(r'D:\sync\brave_data\bad reweard function\episode_return.csv', 124716)