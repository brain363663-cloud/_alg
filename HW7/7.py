def solve_maze(maze, start, end):
    path = []
    rows = len(maze)
    cols = len(maze[0])

    def dfs(x, y):
        # 檢查是否超出邊界或是撞牆/已走過
        if x < 0 or x >= rows or y < 0 or y >= cols or maze[x][y] != 0:
            return False
        
        # 記錄當前位置
        path.append((x, y))
        maze[x][y] = 2  # 標記為已造訪
        
        # 檢查是否到達終點
        if (x, y) == end:
            return True
        
        # 往四個方向探索：上、下、左、右
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            if dfs(x + dx, y + dy):
                return True
        
        # 如果此路不通，回溯：從路徑移除並恢復標記（可選，看是否要顯示最終路徑）
        path.pop()
        return False

    if dfs(start[0], start[1]):
        return path
    else:
        return "找不到出口"

# --- 測試資料 ---
# 0: 路, 1: 牆
maze_map = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start_pos = (0, 0)
end_pos = (4, 4)

result = solve_maze(maze_map, start_pos, end_pos)
print(f"逃脫路徑: {result}")
