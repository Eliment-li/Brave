from collections import deque


def bfs_route(matrix, start_row, start_col, target_values,broken=[]):
    """
    在二维矩阵中查找从起点到任意目标值的最短路径

    参数:
        matrix: 二维矩阵 (list of lists)
        start_row, start_col: 起点坐标
        target_values: 目标值集合 (set)

    返回:
        {
            "path": 路径坐标列表 [(row, col), ...],
            "distance": 路径长度 (步数),
            "target_reached": 到达的目标值
        } 或 None (无路径)
    """
    # 验证输入
    # if (not matrix) or (not matrix[0]):
    #     return None

    rows, cols = len(matrix), len(matrix[0])

    # 检查起点是否有效
    if not (0 <= start_row < rows and 0 <= start_col < cols):
        return None

    # 如果起点就是目标节点
    if matrix[start_row][start_col] in target_values:
        return {
            "path": [(start_row, start_col)],
            "distance": 0,
            "target_reached": matrix[start_row][start_col]
        }

    # 方向数组：上、右、下、左
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # 初始化数据结构
    visited = [[False] * cols for _ in range(rows)]  # 访问标记
    prev = [[None] * cols for _ in range(rows)]  # 前驱节点
    queue = deque()  # BFS队列

    # 起点入队
    queue.append((start_row, start_col))
    visited[start_row][start_col] = True

    # BFS主循环
    found_target = None
    while queue:
        r, c = queue.popleft()

        # 检查当前节点是否为目标节点
        if matrix[r][c] in target_values:
            found_target = (r, c)
            break

        #broken=[(0,1),(1,1),(2,1)]

        # 遍历四个方向
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) in broken:
                continue
            # 检查新位置是否有效
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                visited[nr][nc] = True
                prev[nr][nc] = (r, c)  # 记录前驱节点
                queue.append((nr, nc))

    # 未找到目标节点
    if not found_target:
        return None,None,target_values

    # 回溯构建路径
    path = []
    r, c = found_target
    while (r, c) != (start_row, start_col):
        path.append((r, c))
        r, c = prev[r][c]
    path.append((start_row, start_col))
    path.reverse()  # 反转得到从起点到目标的顺序

    distance = len(path) - 1  # 步数=节点数-1
    # return {
    #     "path": path,
    #     "distance": len(path) - 1,  # 步数=节点数-1
    #     "target_reached": matrix[found_target[0]][found_target[1]]
    # }
    return path,distance,matrix[found_target[0]][found_target[1]]


if __name__ == '__main__':
    # 示例矩阵
    grid = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    # 起点 (0,0) 值=1
    # 目标值集合 {7, 11, 15}
    path,distance,target = bfs_route(
        matrix=grid,
        start_row=0,
        start_col=0,
        target_values={23}
    )

    print(path)
