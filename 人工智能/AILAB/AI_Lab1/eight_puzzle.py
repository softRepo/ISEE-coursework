import heapq
import copy
import random

# 定义目标状态中各个数字的正确坐标 (行, 列)
# 注意：题目要求忽略数字 5，0 是空格，所以字典中严格排除了 0 和 5
TARGET_POS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0),            6: (1, 2),
    7: (2, 0), 8: (2, 1)
}

class PuzzleNode:
    def __init__(self, board, parent=None, move="", g=0, h=0):
        self.board = board      # 当前棋盘状态 (3x3 嵌套列表)
        self.parent = parent    # 父节点，用于回溯最终的求解路径
        self.move = move        # 从父节点到达当前节点，空格移动的方向
        self.g = g              # g(n): 从初始状态到当前状态的实际移动步数
        self.h = h              # h(n): 启发式函数评估的到达目标状态的估计代价
        self.f = g + h          # f(n) = g(n) + h(n): 综合评估代价

    # 重载小于运算符，使得 heapq 优先队列可以根据 f(n) 的值对节点进行排序
    # f(n) 越小，说明该节点越有可能是最优解路径上的节点，优先级越高
    def __lt__(self, other):
        return self.f < other.f

    # 将二维列表形式的棋盘转换为不可变的元组，以便作为集合(Set)的键用于状态判重
    def get_state_tuple(self):
        return tuple(tuple(row) for row in self.board)

def find_zero(board):
    """找到空格(数字0)的坐标"""
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j

def is_goal(board):
    """判断当前棋盘是否达到目标状态（只检查 1,2,3,4,6,7,8）"""
    for i in range(3):
        for j in range(3):
            val = board[i][j]
            if val in TARGET_POS:
                if (i, j) != TARGET_POS[val]:
                    return False
    return True

# ================= 启发式函数定义 =================

def h1_misplaced_tiles(board):
    """启发式函数 1: 错位数字的数量 (忽略 0 和 5)"""
    count = 0
    for i in range(3):
        for j in range(3):
            val = board[i][j]
            # 如果数字在关注列表中，且当前坐标不等于目标坐标，则错位数量+1
            if val in TARGET_POS and (i, j) != TARGET_POS[val]:
                count += 1
    return count

def h2_manhattan_distance(board):
    """启发式函数 2: 曼哈顿距离之和 (忽略 0 和 5)"""
    distance = 0
    for i in range(3):
        for j in range(3):
            val = board[i][j]
            if val in TARGET_POS:
                target_i, target_j = TARGET_POS[val]
                # 计算当前横纵坐标与目标横纵坐标的绝对差值之和
                distance += abs(i - target_i) + abs(j - target_j)
    return distance

# ================= A* 搜索核心逻辑 =================

def get_neighbors(node, heuristic_func):
    """生成当前节点所有合法的下一步状态"""
    neighbors = []
    zx, zy = find_zero(node.board)
    # 定义空格可以移动的四个方向：上、下、左、右
    directions = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}

    for move_name, (dx, dy) in directions.items():
        nx, ny = zx + dx, zy + dy
        # 如果移动后的坐标没有越界
        if 0 <= nx < 3 and 0 <= ny < 3:
            # 深拷贝当前棋盘，防止修改原节点数据
            new_board = copy.deepcopy(node.board)
            # 交换空格和目标方向数字的位置
            new_board[zx][zy], new_board[nx][ny] = new_board[nx][ny], new_board[zx][zy]

            # 实际步数 g(n) 增加 1
            new_g = node.g + 1
            # 计算新状态的启发式代价 h(n)
            new_h = heuristic_func(new_board)
            # 实例化新节点
            new_node = PuzzleNode(new_board, node, move_name, new_g, new_h)
            neighbors.append(new_node)

    return neighbors

def a_star_search(initial_board, heuristic_func):
    """A* 搜索算法主函数"""
    # 1. 初始化起始节点，计算初始的 h(n)，此时 g(n)=0
    start_node = PuzzleNode(initial_board, h=heuristic_func(initial_board))

    # 2. 初始化 Open 表 (优先队列) 和 Closed 表 (集合)
    open_set = []
    heapq.heappush(open_set, start_node)
    closed_set = set() # 用于记录已经访问过的状态，防止算法陷入死循环

    # 3. 开始循环搜索
    while open_set:
        # 从 Open 表中弹出 f(n) 最小的节点作为当前扩展节点
        current_node = heapq.heappop(open_set)

        # 如果当前节点是目标状态，搜索成功，返回该节点以便回溯路径
        if is_goal(current_node.board):
            return current_node

        state_tuple = current_node.get_state_tuple()
        # 如果该状态已经存在于 Closed 表中，说明之前已经找到过更优或相同的路径到达此状态，直接跳过
        if state_tuple in closed_set:
            continue

        # 将当前状态加入 Closed 表
        closed_set.add(state_tuple)

        # 生成并遍历所有合法的邻居节点
        for neighbor in get_neighbors(current_node, heuristic_func):
            # 只有当邻居状态不在 Closed 表中时，才将其加入 Open 表继续探索
            if neighbor.get_state_tuple() not in closed_set:
                heapq.heappush(open_set, neighbor)

    return None # 如果 Open 表为空仍未找到目标，说明无解

# ================= 辅助函数 =================

def generate_solvable_state(steps=30):
    """通过从目标状态随机合法移动，生成一个必定有解的随机初始状态"""
    board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    zx, zy = 2, 2
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(steps):
        valid_moves = []
        for dx, dy in directions:
            if 0 <= zx + dx < 3 and 0 <= zy + dy < 3:
                valid_moves.append((dx, dy))

        dx, dy = random.choice(valid_moves)
        nx, ny = zx + dx, zy + dy
        board[zx][zy], board[nx][ny] = board[nx][ny], board[zx][zy]
        zx, zy = nx, ny

    return board

def print_path(node):
    """回溯并打印完整的求解路径"""
    path = []
    current = node
    while current:
        path.append(current)
        current = current.parent
    path.reverse() # 从终点回溯到起点，需要反转列表

    print(f"\n✅ 成功找到解！总共需要移动 {len(path) - 1} 步。\n")
    for step, n in enumerate(path):
        if step == 0:
            print("初始状态:")
        else:
            print(f"第 {step} 步 (空格向【{n.move}】移动):")

        for row in n.board:
            print(row)
        print("-" * 15)

# ================= 运行测试 =================
if __name__ == "__main__":
    # 随机生成一个保证有解的初始状态
    initial_state = generate_solvable_state(steps=20)

    print("========================================")
    print("🚀 使用启发式函数 2 (曼哈顿距离) 求解中...")
    print("========================================")

    # 调用 A* 搜索，可以通过修改 heuristic_func 参数切换启发式函数
    # 选项: h1_misplaced_tiles 或 h2_manhattan_distance
    result_node = a_star_search(initial_state, heuristic_func=h2_manhattan_distance)

    if result_node:
        print_path(result_node)
    else:
        print("❌ 搜索失败！")
