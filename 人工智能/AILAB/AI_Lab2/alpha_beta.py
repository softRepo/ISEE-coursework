import customtkinter as ctk
from tkinter import messagebox
import random

# ================= 核心算法与启发式评估 =================

def check_winner(board):
    for i in range(4):
        if board[i][0] == board[i][1] == board[i][2] == board[i][3] != ' ': return board[i][0]
    for i in range(4):
        if board[0][i] == board[1][i] == board[2][i] == board[3][i] != ' ': return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] == board[3][3] != ' ': return board[0][0]
    if board[0][3] == board[1][2] == board[2][1] == board[3][0] != ' ': return board[0][3]
    return None

def is_inevitable_draw(board):
    lines = []
    for i in range(4): lines.append([board[i][j] for j in range(4)])
    for j in range(4): lines.append([board[i][j] for i in range(4)])
    lines.append([board[i][i] for i in range(4)])
    lines.append([board[i][3-i] for i in range(4)])
    for line in lines:
        if not ('X' in line and 'O' in line):
            return False
    return True

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

def alpha_beta(board, depth, alpha, beta, is_maximizing, ai_role):
    winner = check_winner(board)
    human_role = 'O' if ai_role == 'X' else 'X'

    if winner == ai_role: return 10000 + depth
    if winner == human_role: return -10000 - depth
    if is_board_full(board) or is_inevitable_draw(board): return 0
    if depth == 0: return evaluate(board, ai_role)

    if is_maximizing:
        best_val = -float('inf')
        for i in range(4):
            for j in range(4):
                if board[i][j] == ' ':
                    board[i][j] = ai_role
                    val = alpha_beta(board, depth - 1, alpha, beta, False, ai_role)
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
                    val = alpha_beta(board, depth - 1, alpha, beta, True, ai_role)
                    board[i][j] = ' '
                    best_val = min(best_val, val)
                    beta = min(beta, val)
                    if beta <= alpha: break
            if beta <= alpha: break
        return best_val

def get_best_move_ab(board, ai_role, depth_limit=4):
    best_score = -float('inf')
    best_move = None
    alpha, beta = -float('inf'), float('inf')
    for i in range(4):
        for j in range(4):
            if board[i][j] == ' ':
                board[i][j] = ai_role
                score = alpha_beta(board, depth_limit - 1, alpha, beta, False, ai_role)
                board[i][j] = ' '
                if score > best_score:
                    best_score, best_move = score, (i, j)
                alpha = max(alpha, best_score)
    return best_move

# ================= 交互界面与状态机控制 =================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
DEFAULT_GRID_COLOR = "#333333"

class TicTacToeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("井字棋 4x4 专业版 - 状态空间搜索")
        self.geometry("950x850")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.difficulty = 4

        self.main_menu = MainMenu(self, self)
        self.game_board = GameFrame(self, self)
        self.show_page("menu")

    def show_page(self, page_name):
        self.main_menu.grid_forget()
        self.game_board.grid_forget()
        if page_name == "menu":
            self.main_menu.grid(row=0, column=0, sticky="nsew")
        elif page_name == "game":
            self.game_board.grid(row=0, column=0, sticky="nsew")
            self.game_board.start_game()

class MainMenu(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        ctk.CTkLabel(self, text="4x4 井字棋与博弈论", font=("Helvetica", 42, "bold")).pack(pady=(100, 20))

        rule_box = ctk.CTkTextbox(self, width=420, height=160, font=("Helvetica", 14), fg_color="#2B2B2B")
        rule_box.pack(pady=20)
        rule_box.insert("0.0", "【实验核心要求】\n1. 四子连珠判定胜负。\n2. 动态Pie规则：先手落一子后，后手可掠夺该棋并互换身份。\n3. 高效剪枝：Alpha-Beta 辅以局部评分启发式函数。\n4. 死锁检测：棋局丧失所有连珠可能性时，提前结算平局。")
        rule_box.configure(state="disabled")

        ctk.CTkLabel(self, text="选择对抗难度 (启发式搜索向下探索层数):").pack(pady=10)
        self.diff_btn = ctk.CTkSegmentedButton(self, values=["新手", "普通", "大师"], command=self.set_diff)
        self.diff_btn.set("普通")
        self.diff_btn.pack(pady=10)

        ctk.CTkButton(self, text="初始化对弈环境", width=200, height=50, command=lambda: controller.show_page("game")).pack(pady=10)
        ctk.CTkButton(self, text="退出", fg_color="#E94E77", command=self.quit).pack(pady=10)

    def set_diff(self, val):
        self.controller.difficulty = {"新手":2, "普通":4, "大师":6}[val]

class GameFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.pack(side="left", padx=30, fill="both", expand=True)

        self.info_label = ctk.CTkLabel(self.left_panel, text="数据流加载中...", font=("Helvetica", 22, "bold"))
        self.info_label.pack(pady=(30, 10))

        self.grid_box = ctk.CTkFrame(self.left_panel)
        self.grid_box.pack(pady=10)
        self.btns = [[None for _ in range(4)] for _ in range(4)]

        for r in range(4):
            for c in range(4):
                btn = ctk.CTkButton(self.grid_box, text="", width=90, height=90, font=("Arial", 30, "bold"),
                                    fg_color=DEFAULT_GRID_COLOR, text_color_disabled="white",
                                    command=lambda row=r, col=c: self.on_click(row, col))
                btn.grid(row=r, column=c, padx=5, pady=5)
                self.btns[r][c] = btn

        self.pie_area = ctk.CTkFrame(self.left_panel, fg_color="transparent", height=60)
        self.pie_area.pack(pady=10)

        # ====== 底部操作按钮区（重新开始与返回菜单） ======
        self.bottom_btn_area = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.bottom_btn_area.pack(side="bottom", pady=20)

        ctk.CTkButton(self.bottom_btn_area, text="🎮 重新开始本局", fg_color="#27AE60", hover_color="#2ECC71",
                      command=self.start_game).pack(side="left", padx=10)
        ctk.CTkButton(self.bottom_btn_area, text="🏠 返回主菜单", fg_color="#6C7A89", hover_color="#34495E",
                      command=lambda: controller.show_page("menu")).pack(side="left", padx=10)

        self.right_panel = ctk.CTkFrame(self, width=320)
        self.right_panel.pack(side="right", fill="y", padx=(0, 20), pady=20)
        ctk.CTkLabel(self.right_panel, text="对抗推演日志", font=("Helvetica", 18, "bold"), text_color="#50E3C2").pack(pady=10)

        self.log_box = ctk.CTkTextbox(self.right_panel, width=280, height=600, font=("Consolas", 13), fg_color="#1A1A1A")
        self.log_box.pack(padx=10, pady=10)

    def log(self, text):
        self.log_box.insert("end", f"▶ {text}\n")
        self.log_box.see("end")

    def start_game(self):
        self.board = [[' ' for _ in range(4)] for _ in range(4)]
        self.log_box.delete("0.0", "end")

        for r in range(4):
            for c in range(4):
                self.btns[r][c].configure(text="", state="normal", fg_color=DEFAULT_GRID_COLOR)

        self.p1 = random.choice(['Human', 'AI'])
        self.p2 = 'AI' if self.p1 == 'Human' else 'Human'
        self.roles = {self.p1: 'X', self.p2: 'O'}
        self.turn = self.p1
        self.stage = 'first'

        self.log("="*20)
        self.log(f"[实验初始化] 最大搜索深度: {self.controller.difficulty}")
        self.log(f"[阵营分配] 先手执X: {'玩家' if self.p1 == 'Human' else 'AI'}")
        self.log(f"[阵营分配] 后手执O: {'玩家' if self.p2 == 'Human' else 'AI'}")
        self.log("等待初盘落子...")
        self.log("="*20)

        self.info_label.configure(text=f"系统随机分配先手：{self.p1} 执 X")

        # 强制清理可能残留的 Pie 规则按钮
        for w in self.pie_area.winfo_children(): w.destroy()

        if self.p1 == 'AI': self.after(600, self.ai_first)

    def ai_first(self):
        move = random.choice([(1,1), (1,2), (2,1), (2,2)])
        self.place(move[0], move[1], 'X')
        self.init_pie()

    def on_click(self, r, c):
        if self.stage == 'over' or self.turn == 'AI': return
        if self.board[r][c] != ' ': return

        if self.stage == 'first':
            self.place(r, c, 'X')
            self.init_pie()
        elif self.stage == 'main':
            self.place(r, c, self.roles['Human'])
            if not self.check_end():
                self.turn = 'AI'
                self.info_label.configure(text="AI 运算推演中...")
                self.after(100, self.ai_main)

    def place(self, r, c, char):
        self.board[r][c] = char
        color = "#E94E77" if char == 'X' else "#4A90E2"
        self.btns[r][c].configure(text=char, text_color="white", text_color_disabled="white", fg_color=color, state="disabled")

        player_name = "AI" if self.roles['AI'] == char else "玩家"
        self.log(f"指令执行: {player_name}(执{char}) 占据坐标 ({r}, {c})")

    def init_pie(self):
        self.stage = 'pie'
        self.info_label.configure(text=f"[Pie规则校验阶段] 等待 {self.p2} 决策")
        self.log("\n[系统判定] 触发 Pie 规则博弈机制")
        if self.p2 == 'Human':
            self.log("[玩家选项] 请在界面决定是否夺取该棋")
            ctk.CTkButton(self.pie_area, text="夺取这步棋 (Swap)", fg_color="#F39C12",
                          command=lambda: self.do_pie(True)).pack(side="left", padx=10)
            ctk.CTkButton(self.pie_area, text="不交换 (正常继续)", fg_color="#27AE60",
                          command=lambda: self.do_pie(False)).pack(side="left", padx=10)
        else:
            self.log("AI 正在评估先手落子的全局威胁度...")
            self.after(1200, self.ai_pie)

    def ai_pie(self):
        fm = [(i,j) for i in range(4) for j in range(4) if self.board[i][j]=='X'][0]
        swap = fm in [(1,1), (1,2), (2,1), (2,2)]
        self.do_pie(swap)

    def do_pie(self, swap):
        for w in self.pie_area.winfo_children(): w.destroy()
        if swap:
            self.roles[self.p1], self.roles[self.p2] = 'O', 'X'
            self.turn = self.p1
            self.log(f"-> 博弈决策: 执行掠夺！")
            self.log(f"-> 阵营反转: 玩家执 {self.roles['Human']}，AI执 {self.roles['AI']}")
        else:
            self.turn = self.p2
            self.log(f"-> 博弈决策: 维持原状态。")

        self.stage = 'main'
        self.log("-"*20)
        self.info_label.configure(text=f"你的回合，你目前执 {self.roles['Human']}")
        if self.turn == 'AI': self.ai_main()

    def ai_main(self):
        move = get_best_move_ab(self.board, self.roles['AI'], self.controller.difficulty)
        if move: self.place(move[0], move[1], self.roles['AI'])
        if not self.check_end():
            self.turn = 'Human'
            self.info_label.configure(text=f"你的回合，你目前执 {self.roles['Human']}")

    def check_end(self):
        w = check_winner(self.board)
        forced_draw = is_inevitable_draw(self.board)
        full = is_board_full(self.board)

        if w or forced_draw or full:
            self.stage = 'over'
            if w:
                res = "验证通过：玩家取得最终胜利" if w == self.roles['Human'] else "推演结束：AI 取得绝对优势"
            elif forced_draw:
                res = "死锁侦测：双方均无法获胜，提前中止判定平局"
            else:
                res = "棋盘空间耗尽：未分胜负"

            self.log(f"\n[终局状态] {res}")
            self.info_label.configure(text=res)
            messagebox.showinfo("实验数据结算", res)
            return True
        return False

if __name__ == "__main__":
    app = TicTacToeApp()
    app.mainloop()
