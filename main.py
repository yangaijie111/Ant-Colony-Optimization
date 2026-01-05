import numpy as np
import random
import sys

# -----------------------------
# 1. 模拟数据生成与参数设置
# -----------------------------
# 假设有 8 个节点：0:入口, 1-4:取餐窗口, 5-7:就餐区, 8-9:通道节点
num_nodes = 10
num_windows = 4  # 窗口数量 (节点1-4)
entrance = 0
dining_areas = [5, 6, 7]  # 就餐区节点列表

# 随机生成节点坐标 (实际应用中应为食堂的真实坐标)
np.random.seed(42)
coordinates = np.random.rand(num_nodes, 2) * 100


# 计算距离矩阵
def euclidean_distance(i, j):
    return np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))


# 初始化距离矩阵
dist_matrix = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        dist_matrix[i][j] = euclidean_distance(i, j)

# 模拟窗口排队时间 (初始化为随机值，算法中会动态变化)
queue_time = np.random.randint(5, 30, size=num_windows)  # 窗口0-3对应节点1-4

# -----------------------------
# 2. 蚁群算法参数
# -----------------------------
num_ants = 10  # 蚂蚁数量
num_iterations = 50  # 迭代次数

alpha = 1  # 信息素重要程度
beta = 2  # 启发函数重要程度
rho = 0.5  # 信息素挥发率
Q = 100  # 信息素常数

# 初始化信息素矩阵 (节点间的信息素浓度)
pheromone = np.ones((num_nodes, num_nodes)) * 0.1


# -----------------------------
# 3. 核心算法实现
# -----------------------------

def solve_aco():
    global pheromone
    best_global_path = None
    best_global_cost = float('inf')

    for it in range(num_iterations):
        all_ant_paths = []
        all_ant_costs = []

        # --- 每只蚂蚁构建路径 ---
        for ant in range(num_ants):
            current_pos = entrance
            path = [entrance]
            visited = set([entrance])
            total_cost = 0.0
            has_taken_food = False  # 标记是否已取餐

            # 路径构建循环
            while not has_taken_food or current_pos not in dining_areas:
                # 1. 确定候选节点 (根据是否已取餐进行约束)
                candidates = []
                for next_node in range(num_nodes):
                    if next_node in visited:
                        continue
                    # 规则1: 未取餐时，只能去窗口或通道
                    if not has_taken_food:
                        if (1 <= next_node <= 4) or (next_node >= 8):
                            candidates.append(next_node)
                    # 规则2: 已取餐时，只能去就餐区或通道
                    else:
                        if next_node in dining_areas or (next_node >= 8):
                            candidates.append(next_node)

                if not candidates:
                    break  # 无路可走，跳出

                # 2. 计算转移概率 (核心公式)
                probabilities = []
                total_prob = 0.0
                for next_node in candidates:
                    # 启发值 eta = 1 / (距离 + 等待时间惩罚)
                    distance = dist_matrix[current_pos][next_node]
                    # 如果是去取餐窗口，加上该窗口的排队时间作为启发值的一部分
                    if 1 <= next_node <= 4:
                        eta = 1.0 / (distance + queue_time[next_node - 1])
                    else:
                        eta = 1.0 / distance

                    # 概率计算公式: (tau^alpha) * (eta^beta)
                    prob = (pheromone[current_pos][next_node] ** alpha) * (eta ** beta)
                    probabilities.append(prob)
                    total_prob += prob

                # 归一化概率
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                else:
                    probabilities = [1.0 / len(probabilities)] * len(probabilities)

                # 3. 轮盘赌选择下一个节点
                rand = random.random()
                select_prob = 0.0
                next_idx = 0
                for i, p in enumerate(probabilities):
                    select_prob += p
                    if rand <= select_prob:
                        next_idx = i
                        break
                next_node = candidates[next_idx]

                # 4. 移动蚂蚁
                path.append(next_node)
                visited.add(next_node)

                # 如果到达窗口，标记已取餐 (仅一次)
                if 1 <= next_node <= 4 and not has_taken_food:
                    has_taken_food = True

                # 累计成本 (距离 + 若是第一次去窗口则加上排队时间)
                travel_cost = dist_matrix[current_pos][next_node]
                wait_cost = queue_time[next_node - 1] if (1 <= next_node <= 4 and next_node not in visited) else 0
                total_cost += travel_cost + wait_cost

                current_pos = next_node

            # --- 单只蚂蚁路径构建结束 ---
            all_ant_paths.append(path)
            all_ant_costs.append(total_cost)

            # 更新全局最优
            if total_cost < best_global_cost:
                best_global_cost = total_cost
                best_global_path = path.copy()

        # --- 信息素更新 (迭代结束后) ---
        # 1. 信息素挥发
        pheromone *= (1 - rho)

        # 2. 信息素增强 (仅由本次迭代最优蚂蚁释放)
        # (也可以改为所有蚂蚁释放，这里为了简单只用本次迭代的最优蚂蚁)
        it_best_cost = min(all_ant_costs)
        it_best_path = all_ant_paths[np.argmin(all_ant_costs)]

        for i in range(len(it_best_path) - 1):
            from_node = it_best_path[i]
            to_node = it_best_path[i + 1]
            if from_node != to_node:
                pheromone[from_node][to_node] += Q / it_best_cost
                pheromone[to_node][from_node] += Q / it_best_cost  # 对称更新

        # 可选：打印当前迭代进度
        # print(f"Iteration {it+1}, Best Cost: {best_global_cost:.2f}")

    return best_global_path, best_global_cost


# -----------------------------
# 4. 运行程序
# -----------------------------
if __name__ == "__main__":
    print("正在优化校园食堂就餐路径...")
    optimal_path, min_cost = solve_aco()

    print("\n--- 优化结果 ---")
    print(f"最优路径: {optimal_path}")
    print(f"路径节点含义: 0=入口, 1-4=取餐窗口, 5-7=就餐区, 8-9=通道")
    print(f"预估最小总成本 (时间): {min_cost:.2f} 单位")

    # 简单判断选择了哪个窗口
    chosen_window = None
    for node in optimal_path:
        if 1 <= node <= 4:
            chosen_window = node
            break
    if chosen_window:
        print(f"建议选择取餐窗口: {chr(64 + chosen_window)} (节点 {chosen_window})")