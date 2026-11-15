import heapq
import copy
import math

TARGET_POS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0),            6: (1, 2),
    7: (2, 0), 8: (2, 1)
}

class PuzzleNode:
    def __init__(self, board, parent=None, move="", g=0, h=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

    def get_state_tuple(self):
        return tuple(tuple(row) for row in self.board)

def find_zero(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j

def is_goal(board):
    for i in range(3):
        for j in range(3):
            val = board[i][j]
            if val in TARGET_POS:
                if (i, j) != TARGET_POS[val]:
                    return False
    return True

# ================= 启发式函数集合 =================

def h1_misplaced_tiles(board):
    """h1: 错位数字的数量"""
    count = 0
    for i in range(3):
        for j in range(3):
            val = board[i][j]
            if val in TARGET_POS and (i, j) != TARGET_POS[val]:
                count += 1
    return count

def h2_manhattan_distance(board):
    """h2: 曼哈顿距离之和"""
    distance = 0
    for i in range(3):
        for j in range(3):
            val = board[i][j]
            if val in TARGET_POS:
                target_i, target_j = TARGET_POS[val]
                distance += abs(i - target_i) + abs(j - target_j)
    return distance

def h3_euclidean_distance(board):
    """h3: 欧几里得直线距离之和"""
    distance = 0.0
    for i in range(3):
        for j in range(3):
            val = board[i][j]
            if val in TARGET_POS:
                target_i, target_j = TARGET_POS[val]
                # 计算两点之间的直线距离
                distance += math.sqrt((i - target_i)**2 + (j - target_j)**2)
    return distance

# ================= A* 搜索核心逻辑 =================

def get_neighbors(node, heuristic_func):
    neighbors = []
    zx, zy = find_zero(node.board)
    directions = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}

    for move_name, (dx, dy) in directions.items():
        nx, ny = zx + dx, zy + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_board = copy.deepcopy(node.board)
            new_board[zx][zy], new_board[nx][ny] = new_board[nx][ny], new_board[zx][zy]

            new_g = node.g + 1
            new_h = heuristic_func(new_board)
            new_node = PuzzleNode(new_board, node, move_name, new_g, new_h)
            neighbors.append(new_node)

    return neighbors

def a_star_search(initial_board, heuristic_func):
    """
    修改了返回值，现在同时返回: (找到的最终节点, 算法搜索过程中探索的节点总数)
    """
    start_node = PuzzleNode(initial_board, h=heuristic_func(initial_board))
    open_set = []
    heapq.heappush(open_set, start_node)
    closed_set = set()

    expanded_nodes = 0 # 记录探索了多少个节点

    while open_set:
        current_node = heapq.heappop(open_set)
        expanded_nodes += 1

        if is_goal(current_node.board):
            return current_node, expanded_nodes

        state_tuple = current_node.get_state_tuple()
        if state_tuple in closed_set:
            continue

        closed_set.add(state_tuple)

        for neighbor in get_neighbors(current_node, heuristic_func):
            if neighbor.get_state_tuple() not in closed_set:
                heapq.heappush(open_set, neighbor)

    return None, expanded_nodes

def get_path_details(node):
    """只提取路径，不打印（交给主函数排版）"""
    path = []
    current = node
    while current:
        path.append(current)
        current = current.parent
    path.reverse()
    return path

def print_single_step(node):
    """打印单个棋盘状态"""
    for row in node.board:
        print(row)
    print("-" * 15)

# ================= 自动化测试与对比 =================
if __name__ == "__main__":
    # 使用之前成功跑通的初始状态，确保结果一致
    initial_state = [
        [1, 6, 2],
        [4, 0, 3],
        [7, 5, 8]
    ]

    # 定义要测试的启发式函数列表
    heuristics_to_test = [
        ("h1 (错位数字数量)", h1_misplaced_tiles),
        ("h2 (曼哈顿距离)", h2_manhattan_distance),
        ("h3 (欧几里得距离)", h3_euclidean_distance)
    ]

    print("初始状态为:")
    for row in initial_state:
        print(row)
    print("=" * 40)

    for name, func in heuristics_to_test:
        print(f"\n🚀 开始使用【{name}】进行求解...")
        result_node, expanded_count = a_star_search(initial_state, heuristic_func=func)

        if result_node:
            path = get_path_details(result_node)
            total_steps = len(path) - 1
            print(f"✅ 求解成功！")
            print(f"📊 性能统计：寻找该解共探索了 {expanded_count} 个状态节点。")
            print(f"🚶‍♂️ 总移动步数：{total_steps} 步。")

            print("\n--- 详细移动路径 ---")
            for step, n in enumerate(path):
                if step > 0:
                        print(f"第 {step} 步 (空格向【{n.move}】移动):")
                        print_single_step(n)
        else:
            print("❌ 求解失败！")
        print("=" * 40)
