import time

# 全局变量，用于统计评估的节点数
node_count = 0

# ================= 基础判断与评估函数 =================
def check_winner(board):
    for i in range(4):
        if board[i][0] == board[i][1] == board[i][2] == board[i][3] != ' ': return board[i][0]
    for i in range(4):
        if board[0][i] == board[1][i] == board[2][i] == board[3][i] != ' ': return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] == board[3][3] != ' ': return board[0][0]
    if board[0][3] == board[1][2] == board[2][1] == board[3][0] != ' ': return board[0][3]
    return None

def is_board_full(board):
    for row in board:
        if ' ' in row: return False
    return True

def evaluate(board, ai_role):
    human_role = 'O' if ai_role == 'X' else 'X'
    score = 0
    lines = []
    for i in range(4): lines.append([board[i][j] for j in range(4)])
    for j in range(4): lines.append([board[i][j] for i in range(4)])
    lines.append([board[i][i] for i in range(4)])
    lines.append([board[i][3-i] for i in range(4)])

    for line in lines:
        ai_c, hu_c, empty = line.count(ai_role), line.count(human_role), line.count(' ')
        if ai_c > 0 and hu_c > 0: continue
        if ai_c == 4: score += 10000
        elif ai_c == 3 and empty == 1: score += 150
        elif ai_c == 2 and empty == 2: score += 20
        if hu_c == 4: score -= 10000
        elif hu_c == 3 and empty == 1: score -= 150
    return score

# ================= 算法 1：无限制纯 Minimax =================
def minimax_unlimited(board, is_maximizing, ai_role):
    global node_count
    node_count += 1

    winner = check_winner(board)
    human_role = 'O' if ai_role == 'X' else 'X'

    if winner == ai_role: return 10000
    if winner == human_role: return -10000
    if is_board_full(board): return 0

    if is_maximizing:
        best_val = -float('inf')
        for i in range(4):
            for j in range(4):
                if board[i][j] == ' ':
                    board[i][j] = ai_role
                    best_val = max(best_val, minimax_unlimited(board, False, ai_role))
                    board[i][j] = ' '
        return best_val
    else:
        best_val = float('inf')
        for i in range(4):
            for j in range(4):
                if board[i][j] == ' ':
                    board[i][j] = human_role
                    best_val = min(best_val, minimax_unlimited(board, True, ai_role))
                    board[i][j] = ' '
        return best_val

def run_unlimited(board, ai_role):
    global node_count
    node_count = 0
    best_score = -float('inf')
    best_move = None

    start_time = time.time()
    for i in range(4):
        for j in range(4):
            if board[i][j] == ' ':
                board[i][j] = ai_role
                score = minimax_unlimited(board, False, ai_role)
                board[i][j] = ' '
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
    end_time = time.time()
    return best_move, end_time - start_time, node_count

# ================= 算法 2：深度受限 Alpha-Beta 剪枝 =================
def alpha_beta_limited(board, depth, alpha, beta, is_maximizing, ai_role):
    global node_count
    node_count += 1

    winner = check_winner(board)
    human_role = 'O' if ai_role == 'X' else 'X'

    if winner == ai_role: return 10000 + depth
    if winner == human_role: return -10000 - depth
    if is_board_full(board): return 0
    if depth == 0: return evaluate(board, ai_role)

    if is_maximizing:
        best_val = -float('inf')
        for i in range(4):
            for j in range(4):
                if board[i][j] == ' ':
                    board[i][j] = ai_role
                    val = alpha_beta_limited(board, depth - 1, alpha, beta, False, ai_role)
                    board[i][j] = ' '
                    best_val = max(best_val, val)
                    alpha = max(alpha, val)
                    if beta <= alpha: break
            if beta <= alpha: break
        return best_val
    else:
        best_val = float('inf')
        for i in range(4):
            for j in range(4):
                if board[i][j] == ' ':
                    board[i][j] = human_role
                    val = alpha_beta_limited(board, depth - 1, alpha, beta, True, ai_role)
                    board[i][j] = ' '
                    best_val = min(best_val, val)
                    beta = min(beta, val)
                    if beta <= alpha: break
            if beta <= alpha: break
        return best_val

def run_limited_ab(board, ai_role, depth_limit):
    global node_count
    node_count = 0
    best_score = -float('inf')
    best_move = None
    alpha, beta = -float('inf'), float('inf')

    start_time = time.time()
    for i in range(4):
        for j in range(4):
            if board[i][j] == ' ':
                board[i][j] = ai_role
                score = alpha_beta_limited(board, depth_limit - 1, alpha, beta, False, ai_role)
                board[i][j] = ' '
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
                alpha = max(alpha, best_score)
    end_time = time.time()
    return best_move, end_time - start_time, node_count

# ================= 主程序：残局对比测试 =================
if __name__ == "__main__":
    # 构造一个 4x4 的残局，剩余 9 个空格。
    # 纯 Minimax 搜索 9 个空格的状态空间约为 9! (36万) 个节点，耗时几秒钟，非常适合用来展示对比。
    endgame_board = [
        ['X', 'O', ' ', 'X'],
        [' ', 'X', 'O', ' '],
        ['O', ' ', ' ', ' '],
        ['X', ' ', ' ', 'O']
    ]

    print("==================================================")
    print("性能测试：同一残局下的算法评估对比")
    print("残局状态：剩余 9 个空位。AI 执 'X'")
    print("==================================================\n")

    # 1. 无限制 Minimax
    print("[1] 正在运行 无限制纯 Minimax 算法...")
    move1, time1, nodes1 = run_unlimited(endgame_board, 'X')
    print(f"    - 最佳落子预测: {move1}")
    print(f"    - 评估节点总数: {nodes1:,} 个")
    print(f"    - 思考总耗时  : {time1:.4f} 秒\n")

    # 2. 深度受限 Alpha-Beta
    depth = 4
    print(f"[2] 正在运行 深度受限 Alpha-Beta 剪枝 (限制深度={depth})...")
    move2, time2, nodes2 = run_limited_ab(endgame_board, 'X', depth)
    print(f"    - 最佳落子预测: {move2}")
    print(f"    - 评估节点总数: {nodes2:,} 个")
    print(f"    - 思考总耗时  : {time2:.4f} 秒\n")

    print("==================================================")
    print(f"性能提升倍数估算：")
    if time2 > 0:
        print(f"    - 速度提升约 {time1/time2:.0f} 倍")
        print(f"    - 节点减少约 {nodes1/nodes2:.0f} 倍")
    print("==================================================")
