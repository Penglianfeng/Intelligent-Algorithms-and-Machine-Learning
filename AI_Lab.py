import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import math
import time
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import urllib.request
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QRadioButton, QLineEdit,
    QGroupBox, QProgressBar, QTabWidget, QFileDialog, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
matplotlib.use("Qt5Agg")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== TSP问题数据集 ====================
# Oliver 30城市
cities_oliver_30 = [
    (87, 7), (91, 38), (83, 46), (71, 44), (64, 60),
    (68, 58), (83, 69), (87, 76), (74, 78), (71, 71),
    (58, 69), (54, 62), (51, 67), (37, 84), (41, 94),
    (2, 99), (7, 64), (22, 60), (25, 62), (18, 54),
    (4, 50), (13, 40), (18, 40), (24, 42), (25, 38),
    (41, 26), (45, 21), (44, 35), (58, 35), (62, 32)
]

# Berlin 52城市 (部分)
cities_berlin_52 = [
    (565, 575), (25, 185), (345, 750), (945, 685), (845, 655),
    (880, 660), (25, 230), (525, 1000), (580, 1175), (650, 1130),
    (1605, 620), (1220, 580), (1465, 200), (1530, 5), (845, 680),
    (725, 370), (145, 665), (415, 635), (510, 875), (560, 365),
    (300, 465), (520, 585), (480, 415), (835, 625), (975, 580),
    (1215, 245), (1320, 315), (1250, 400), (660, 180), (410, 250),
    (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595),
    (685, 610), (770, 610), (795, 645), (720, 635), (760, 650),
    (475, 960), (95, 260), (875, 920), (700, 500), (555, 815),
    (830, 485), (1170, 65), (830, 610), (605, 625), (595, 360),
    (1340, 725), (1740, 245)
]

# Eil 51城市 (部分)
cities_eil_51 = [
    (37, 52), (49, 49), (52, 64), (20, 26), (40, 30),
    (21, 47), (17, 63), (31, 62), (52, 33), (51, 21),
    (42, 41), (31, 32), (5, 25), (12, 42), (36, 16),
    (52, 41), (27, 23), (17, 33), (13, 13), (57, 58),
    (62, 42), (42, 57), (16, 57), (8, 52), (7, 38),
    (27, 68), (30, 48), (43, 67), (58, 48), (58, 27),
    (37, 69), (38, 46), (46, 10), (61, 33), (62, 63),
    (63, 69), (32, 22), (45, 35), (59, 15), (5, 6),
    (10, 17), (21, 10), (5, 64), (30, 15), (39, 10),
    (32, 39), (25, 32), (25, 55), (48, 28), (56, 37),
    (30, 40)
]

# 15城市样例
cities_15 = [
    (30, 40), (37, 52), (49, 49), (52, 64), (31, 62),
    (52, 33), (42, 41), (52, 41), (57, 58), (62, 42),
    (42, 57), (27, 68), (43, 67), (58, 48), (58, 27)
]

# 20城市样例
cities_20 = cities_oliver_30[:20]

# 数据集字典
TSP_DATASETS = {
    "10城市 (Oliver)": (cities_oliver_30[:10], 166.541336),
    "15城市": (cities_15, None),
    "20城市 (Oliver)": (cities_20, None),
    "30城市 (Oliver)": (cities_oliver_30, 424.869292),
    "51城市 (Eil)": (cities_eil_51, 426.0),
    "52城市 (Berlin)": (cities_berlin_52, 7542.0)
}


# ==================== 遗传算法实现 ====================
class GeneticAlgorithmTSP:
    def __init__(self, cities, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500):
        self.cities = cities
        self.num_cities = len(cities)
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []
        self.best_distance = float('inf')
        self.best_path = []
        self.history = []

    def calculate_distance(self, path):
        distance = 0
        for i in range(len(path)):
            city1 = self.cities[path[i]]
            city2 = self.cities[path[(i + 1) % len(path)]]
            distance += math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        return distance

    def create_individual(self):
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def initial_population(self):
        self.population = [self.create_individual() for _ in range(self.pop_size)]

    def rank_population(self):
        ranked = [(self.calculate_distance(ind), ind) for ind in self.population]
        ranked.sort(key=lambda x: x[0])
        return ranked

    def selection(self, ranked_pop):
        selection_results = []
        for i in range(self.elite_size):
            selection_results.append(ranked_pop[i][1])

        df = [1 / (rank[0] + 1e-10) for rank in ranked_pop]
        total_fitness = sum(df)
        probabilities = [f / total_fitness for f in df]

        for _ in range(self.pop_size - self.elite_size):
            pick = random.random()
            current = 0
            for i in range(len(ranked_pop)):
                current += probabilities[i]
                if current > pick:
                    selection_results.append(ranked_pop[i][1])
                    break

        return selection_results

    def crossover(self, parent1, parent2):
        child = [-1] * self.num_cities
        start = random.randint(0, self.num_cities - 1)
        end = random.randint(start, self.num_cities - 1)

        for i in range(start, end + 1):
            child[i] = parent1[i]

        current_pos = 0
        for i in range(self.num_cities):
            if child[i] == -1:
                while parent2[current_pos] in child:
                    current_pos += 1
                child[i] = parent2[current_pos]

        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.num_cities), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def evolve(self, callback=None):
        self.initial_population()

        for generation in range(self.generations):
            ranked = self.rank_population()

            current_best_dist = ranked[0][0]
            current_best_path = ranked[0][1]

            if current_best_dist < self.best_distance:
                self.best_distance = current_best_dist
                self.best_path = current_best_path.copy()

            self.history.append({
                'generation': generation,
                'best_distance': current_best_dist,
                'avg_distance': sum(r[0] for r in ranked) / len(ranked),
                'best_path': current_best_path
            })

            if callback:
                callback(generation, current_best_dist, current_best_path, self.generations)

            selection = self.selection(ranked)

            children = []
            for i in range(0, self.pop_size - self.elite_size, 2):
                parent1 = selection[i]
                parent2 = selection[min(i + 1, len(selection) - 1)]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                children.append(self.mutate(child1))
                children.append(self.mutate(child2))

            elite = [ranked[i][1] for i in range(self.elite_size)]
            self.population = elite + children[:self.pop_size - self.elite_size]

        return self.best_path, self.best_distance


# ==================== 蚁群算法实现 ====================
class AntColonyTSP:
    def __init__(self, cities, num_ants=50, evaporation_rate=0.5, alpha=1, beta=2,
                 q0=0.9, iterations=200):
        self.cities = cities
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.iterations = iterations

        self.dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dx = cities[i][0] - cities[j][0]
                    dy = cities[i][1] - cities[j][1]
                    self.dist_matrix[i][j] = math.sqrt(dx * dx + dy * dy)

        self.pheromone = np.ones((self.num_cities, self.num_cities))
        self.best_path = []
        self.best_distance = float('inf')
        self.history = []

    def calculate_distance(self, path):
        distance = 0
        for i in range(len(path)):
            distance += self.dist_matrix[path[i], path[(i + 1) % len(path)]]
        return distance

    def run(self, callback=None):
        for iteration in range(self.iterations):
            all_paths = []
            all_distances = []

            for _ in range(self.num_ants):
                path = []
                visited = set()
                current = random.randint(0, self.num_cities - 1)
                path.append(current)
                visited.add(current)

                while len(path) < self.num_cities:
                    probabilities = []
                    for next_city in range(self.num_cities):
                        if next_city not in visited:
                            tau = self.pheromone[current, next_city] ** self.alpha
                            eta = (1.0 / (self.dist_matrix[current, next_city] + 1e-10)) ** self.beta
                            probabilities.append(tau * eta)
                        else:
                            probabilities.append(0)

                    total = sum(probabilities)
                    if total > 0:
                        probabilities = [p / total for p in probabilities]

                        if random.random() < self.q0:
                            next_city = int(np.argmax(probabilities))
                        else:
                            next_city = int(np.random.choice(range(self.num_cities), p=probabilities))
                    else:
                        unvisited = [c for c in range(self.num_cities) if c not in visited]
                        next_city = random.choice(unvisited)

                    path.append(next_city)
                    visited.add(next_city)
                    current = next_city

                distance = self.calculate_distance(path)
                all_paths.append(path)
                all_distances.append(distance)

            min_idx = int(np.argmin(all_distances))
            if all_distances[min_idx] < self.best_distance:
                self.best_distance = all_distances[min_idx]
                self.best_path = all_paths[min_idx].copy()

            self.pheromone *= (1 - self.evaporation_rate)

            for i in range(len(self.best_path)):
                city1 = self.best_path[i]
                city2 = self.best_path[(i + 1) % len(self.best_path)]
                delta = 1.0 / (self.best_distance + 1e-10)
                self.pheromone[city1, city2] += delta
                self.pheromone[city2, city1] += delta

            self.history.append({
                'iteration': iteration + 1,
                'best_distance': self.best_distance,
                'avg_distance': float(np.mean(all_distances)),
                'best_path': self.best_path.copy()
            })

            if callback:
                callback(iteration + 1, self.best_distance, self.best_path, self.iterations)

        return self.best_path, self.best_distance


# ==================== 汽车评估数据集加载函数 ====================
def load_car_evaluation():
    """加载UCI汽车评估数据集"""
    # 数据文件路径
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(data_dir, "car.data")
    
    # 如果文件不存在，从UCI下载
    if not os.path.exists(data_file):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        try:
            urllib.request.urlretrieve(url, data_file)
        except Exception as e:
            raise Exception(f"无法下载数据集: {e}")
    
    # 定义列名
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    
    # 读取数据
    df = pd.read_csv(data_file, names=columns)
    
    # 定义特征的类别顺序（用于有序编码）
    category_orders = {
        'buying': ['low', 'med', 'high', 'vhigh'],
        'maint': ['low', 'med', 'high', 'vhigh'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high']
    }
    
    # 目标变量的类别顺序
    class_order = ['unacc', 'acc', 'good', 'vgood']
    
    # 对特征进行编码
    X = df.drop('class', axis=1)
    y = df['class']
    
    # 使用OrdinalEncoder对特征编码
    encoder = OrdinalEncoder(categories=[category_orders[col] for col in X.columns])
    X_encoded = encoder.fit_transform(X)
    
    # 对目标变量编码
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_order)
    y_encoded = label_encoder.transform(y)
    
    return X_encoded, y_encoded, list(X.columns), class_order


# ==================== 机器学习分类器 ====================
class MachineLearningClassifier:
    def __init__(self, dataset_name="iris"):
        self.is_categorical = False  # 标记是否为类别特征数据集
        
        if dataset_name == "iris":
            data = load_iris()
            self.dataset_name = "鸢尾花(Iris)"
            self.X = data.data
            self.y = data.target
            self.feature_names = list(data.feature_names)
            self.target_names = list(data.target_names)
        elif dataset_name == "wine":
            data = load_wine()
            self.dataset_name = "葡萄酒(Wine)"
            self.X = data.data
            self.y = data.target
            self.feature_names = list(data.feature_names)
            self.target_names = list(data.target_names)
        elif dataset_name == "car":
            # 汽车评估数据集
            self.X, self.y, self.feature_names, self.target_names = load_car_evaluation()
            self.dataset_name = "汽车评估(Car Evaluation)"
            self.is_categorical = True
        else:  # breast_cancer
            data = load_breast_cancer()
            self.dataset_name = "乳腺癌(Breast Cancer)"
            self.X = data.data
            self.y = data.target
            self.feature_names = list(data.feature_names)
            self.target_names = list(data.target_names)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

        self.bayes_model = None
        self.tree_model = None
        self.bayes_accuracy = 0
        self.tree_accuracy = 0

    def train_bayes(self):
        # 对于类别特征使用CategoricalNB，对于连续特征使用GaussianNB
        if self.is_categorical:
            self.bayes_model = CategoricalNB()
        else:
            self.bayes_model = GaussianNB()
        self.bayes_model.fit(self.X_train, self.y_train)
        y_pred = self.bayes_model.predict(self.X_test)
        self.bayes_accuracy = accuracy_score(self.y_test, y_pred)
        return classification_report(self.y_test, y_pred, target_names=self.target_names)

    def train_decision_tree(self, max_depth=5):
        # 汽车评估数据集类别较多，使用更深的树
        depth = max_depth if not self.is_categorical else max(max_depth, 5)
        self.tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        self.tree_model.fit(self.X_train, self.y_train)
        y_pred = self.tree_model.predict(self.X_test)
        self.tree_accuracy = accuracy_score(self.y_test, y_pred)
        return classification_report(self.y_test, y_pred, target_names=self.target_names)


# ==================== 算法执行线程 ====================
class AlgorithmThread(QThread):
    update_signal = pyqtSignal(int, float, list)
    finished_signal = pyqtSignal(list, float, float, str)
    progress_signal = pyqtSignal(int, str)

    def __init__(self, algorithm_type, cities, params):
        super().__init__()
        self.algorithm_type = algorithm_type
        self.cities = cities
        self.params = params
        self.history = []

    def run(self):
        start_time = time.time()

        if self.algorithm_type == "遗传算法":
            ga = GeneticAlgorithmTSP(
                cities=self.cities,
                pop_size=self.params['pop_size'],
                elite_size=self.params['elite_size'],
                mutation_rate=self.params['mutation_rate'],
                generations=self.params['generations']
            )

            def callback(generation, distance, path, total):
                if generation % 10 == 0:
                    self.update_signal.emit(generation + 1, distance, path)
                progress = ((generation + 1) / total) * 100
                self.progress_signal.emit(int(progress), f"遗传算法 - 迭代: {generation + 1}/{total}")

            best_path, best_distance = ga.evolve(callback)
            self.history = ga.history

        else:  # 蚁群算法
            aco = AntColonyTSP(
                cities=self.cities,
                num_ants=self.params['num_ants'],
                evaporation_rate=self.params['evaporation_rate'],
                alpha=self.params['alpha'],
                beta=self.params['beta'],
                q0=self.params['q0'],
                iterations=self.params['iterations']
            )

            def callback(iteration, distance, path, total):
                if iteration % 10 == 0:
                    self.update_signal.emit(iteration, distance, path)
                progress = (iteration / total) * 100
                self.progress_signal.emit(int(progress), f"蚁群算法 - 迭代: {iteration}/{total}")

            best_path, best_distance = aco.run(callback)
            self.history = aco.history

        end_time = time.time()
        self.finished_signal.emit(best_path, best_distance, end_time - start_time, self.algorithm_type)


# ==================== 分类实验线程 ====================
class ClassifierThread(QThread):
    finished_signal = pyqtSignal(str, object, list, list)

    def __init__(self, dataset_key):
        super().__init__()
        self.dataset_key = dataset_key

    def run(self):
        try:
            classifier = MachineLearningClassifier(self.dataset_key)

            bayes_report = classifier.train_bayes()
            # 汽车评估数据集使用更深的决策树
            max_tree_depth = 8 if self.dataset_key == "car" else 3
            tree_report = classifier.train_decision_tree(max_depth=max_tree_depth)

            text_lines = []
            text_lines.append("🚀 分类实验完成\n")
            text_lines.append("=" * 70)
            text_lines.append("📊 朴素贝叶斯分类器结果")
            text_lines.append("=" * 70)
            # 说明使用的朴素贝叶斯类型
            if self.dataset_key == "car":
                text_lines.append("📌 算法: CategoricalNB（类别型朴素贝叶斯）")
            else:
                text_lines.append("📌 算法: GaussianNB（高斯朴素贝叶斯）")
            text_lines.append(f"✅ 准确率: {classifier.bayes_accuracy:.4f}")
            text_lines.append("\n📋 分类报告:")
            text_lines.append(bayes_report)

            text_lines.append("\n" + "=" * 70)
            text_lines.append("🌲 决策树分类器结果")
            text_lines.append("=" * 70)
            text_lines.append(f"📌 决策树最大深度: {max_tree_depth}")
            text_lines.append(f"✅ 准确率: {classifier.tree_accuracy:.4f}")
            text_lines.append("\n📋 分类报告:")
            text_lines.append(tree_report)

            text_lines.append("\n" + "=" * 70)
            text_lines.append("📁 数据集信息")
            text_lines.append("=" * 70)
            text_lines.append(f"📦 数据集: {classifier.dataset_name}")
            text_lines.append(f"🔢 特征数: {classifier.X.shape[1]}")
            text_lines.append(f"📊 样本数: {classifier.X.shape[0]}")
            text_lines.append(f"🏷️  类别数: {len(np.unique(classifier.y))}")
            text_lines.append(f"📚 训练集大小: {classifier.X_train.shape[0]}")
            text_lines.append(f"🧪 测试集大小: {classifier.X_test.shape[0]}")
            
            # 汽车评估数据集的额外信息
            if self.dataset_key == "car":
                text_lines.append("\n" + "=" * 70)
                text_lines.append("🚗 汽车评估数据集特征说明 (UCI Machine Learning Repository)")
                text_lines.append("=" * 70)
                text_lines.append("📋 特征变量:")
                text_lines.append("  • buying   : 购买价格 (vhigh, high, med, low)")
                text_lines.append("  • maint    : 维护费用 (vhigh, high, med, low)")
                text_lines.append("  • doors    : 车门数量 (2, 3, 4, 5more)")
                text_lines.append("  • persons  : 载客量   (2, 4, more)")
                text_lines.append("  • lug_boot : 行李箱   (small, med, big)")
                text_lines.append("  • safety   : 安全性   (low, med, high)")
                text_lines.append("\n📋 目标类别 (汽车评估等级):")
                text_lines.append("  • unacc : 不可接受 (Unacceptable)")
                text_lines.append("  • acc   : 可接受   (Acceptable)")
                text_lines.append("  • good  : 良好     (Good)")
                text_lines.append("  • vgood : 非常好   (Very Good)")
                text_lines.append("\n📋 数据集来源:")
                text_lines.append("  URL: https://archive.ics.uci.edu/dataset/19/car+evaluation")

            result_text = "\n".join(text_lines)

            # ✅ 只返回“可序列化信息”，不返回模型对象
            tree_info = {
                "X": classifier.X_train,
                "y": classifier.y_train,
                "max_depth": max_tree_depth
            }

            self.finished_signal.emit(
                result_text,
                tree_info,
                classifier.feature_names,
                classifier.target_names
            )

        except Exception as e:
            self.finished_signal.emit(f"分类实验出错：{str(e)}", None, [], [])



# ==================== 主窗口 ====================
class AIExperimentGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人工智能原理实验四：智能算法与机器学习")
        self.setGeometry(50, 50, 1600, 950)

        self.current_dataset_name = "10城市 (Oliver)"
        self.current_cities, self.optimal_length = TSP_DATASETS[self.current_dataset_name]
        self.algorithm_thread = None
        self.classifier_thread = None
        self.history = []

        self.apply_styles()
        self.setup_ui()

    def apply_styles(self):
        self.setStyleSheet("""
            /* ===== 全局背景 ===== */
            QMainWindow {
                background-color: #f7f9fc;
            }

            QWidget {
                font-family: 'Microsoft YaHei UI';
                font-size: 12px;
                background-color: #f7f9fc;
            }

            /* ===== GroupBox ===== */
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #e1e5eb;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #1f2937;
                font-size: 13px;
                font-weight: bold;
            }

            /* ===== 按钮（Windows 11 Fluent 半扁平） ===== */
            QPushButton {
                background-color: #e8eef7;
                color: #1f2937;
                border: none;
                padding: 10px 14px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #dbe7fb;
            }
            QPushButton:pressed {
                background-color: #c9daf8;
            }
            QPushButton:disabled {
                background-color: #e5e7eb;
                color: #9ca3af;
            }

            /* ===== 输入控件 ===== */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #3b82f6;
                background-color: #ffffff;
            }

            /* ===== 文本框 ===== */
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
            }

            /* ===== TabWidget ===== */
            QTabWidget::pane {
                border: 1px solid #e1e5eb;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #eef2f7;
                color: #374151;
                padding: 10px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #2563eb;
                font-weight: bold;
            }

            /* ===== 进度条 ===== */
            QProgressBar {
                background-color: #e5e7eb;
                border: none;
                border-radius: 6px;
                height: 14px;
                text-align: center;
                color: #1f2937;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 6px;
            }

            /* ===== 标签 ===== */
            QLabel {
                color: #374151;
                font-size: 12px;
            }

            /* ===== 单选按钮 ===== */
            QRadioButton {
                color: #374151;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
        """)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        control_widget = self.create_control_panel()
        main_layout.addWidget(control_widget)

        result_widget = self.create_result_panel()
        main_layout.addWidget(result_widget, stretch=1)

    def create_control_panel(self):
        control_widget = QWidget()
        control_widget.setMaximumWidth(400)
        layout = QVBoxLayout(control_widget)
        layout.setSpacing(12)

        title_label = QLabel("🎯 控制面板")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        layout.addWidget(title_label)

        # TSP数据集选择
        dataset_group = QGroupBox("📊 TSP问题数据集")
        dataset_layout = QVBoxLayout()

        dataset_select_layout = QHBoxLayout()
        dataset_select_layout.addWidget(QLabel("选择数据集:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(list(TSP_DATASETS.keys()))
        self.dataset_combo.currentTextChanged.connect(self.change_dataset)
        dataset_select_layout.addWidget(self.dataset_combo)
        dataset_layout.addLayout(dataset_select_layout)

        self.dataset_info_label = QLabel()
        self.dataset_info_label.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 5px;")
        self.update_dataset_info()
        dataset_layout.addWidget(self.dataset_info_label)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # 算法选择
        algorithm_group = QGroupBox("🔬 选择算法")
        algorithm_layout = QVBoxLayout()
        self.radio_ga = QRadioButton("🧬 遗传算法 (Genetic Algorithm)")
        self.radio_ga.setChecked(True)
        self.radio_aco = QRadioButton("🐜 蚁群算法 (Ant Colony)")
        algorithm_layout.addWidget(self.radio_ga)
        algorithm_layout.addWidget(self.radio_aco)
        algorithm_group.setLayout(algorithm_layout)
        layout.addWidget(algorithm_group)

        # 遗传算法参数
        ga_group = QGroupBox("⚙️ 遗传算法参数")
        ga_layout = QVBoxLayout()

        ga_pop_layout = QHBoxLayout()
        ga_pop_layout.addWidget(QLabel("种群规模:"))
        self.ga_pop_size = QSpinBox()
        self.ga_pop_size.setRange(20, 500)
        self.ga_pop_size.setValue(100)
        self.ga_pop_size.setSingleStep(10)
        ga_pop_layout.addWidget(self.ga_pop_size)
        ga_layout.addLayout(ga_pop_layout)

        ga_mut_layout = QHBoxLayout()
        ga_mut_layout.addWidget(QLabel("变异概率:"))
        self.ga_mutation = QDoubleSpinBox()
        self.ga_mutation.setRange(0.001, 0.5)
        self.ga_mutation.setValue(0.01)
        self.ga_mutation.setSingleStep(0.001)
        self.ga_mutation.setDecimals(3)
        ga_mut_layout.addWidget(self.ga_mutation)
        ga_layout.addLayout(ga_mut_layout)

        ga_gen_layout = QHBoxLayout()
        ga_gen_layout.addWidget(QLabel("迭代次数:"))
        self.ga_generations = QSpinBox()
        self.ga_generations.setRange(100, 2000)
        self.ga_generations.setValue(500)
        self.ga_generations.setSingleStep(50)
        ga_gen_layout.addWidget(self.ga_generations)
        ga_layout.addLayout(ga_gen_layout)

        ga_group.setLayout(ga_layout)
        layout.addWidget(ga_group)

        # 蚁群算法参数
        aco_group = QGroupBox("⚙️ 蚁群算法参数")
        aco_layout = QVBoxLayout()

        aco_ants_layout = QHBoxLayout()
        aco_ants_layout.addWidget(QLabel("蚂蚁数量:"))
        self.aco_num_ants = QSpinBox()
        self.aco_num_ants.setRange(10, 200)
        self.aco_num_ants.setValue(50)
        self.aco_num_ants.setSingleStep(10)
        aco_ants_layout.addWidget(self.aco_num_ants)
        aco_layout.addLayout(aco_ants_layout)

        aco_evap_layout = QHBoxLayout()
        aco_evap_layout.addWidget(QLabel("信息素挥发率:"))
        self.aco_evaporation = QDoubleSpinBox()
        self.aco_evaporation.setRange(0.1, 0.9)
        self.aco_evaporation.setValue(0.5)
        self.aco_evaporation.setSingleStep(0.1)
        self.aco_evaporation.setDecimals(2)
        aco_evap_layout.addWidget(self.aco_evaporation)
        aco_layout.addLayout(aco_evap_layout)

        aco_iter_layout = QHBoxLayout()
        aco_iter_layout.addWidget(QLabel("迭代次数:"))
        self.aco_iterations = QSpinBox()
        self.aco_iterations.setRange(50, 1000)
        self.aco_iterations.setValue(200)
        self.aco_iterations.setSingleStep(50)
        aco_iter_layout.addWidget(self.aco_iterations)
        aco_layout.addLayout(aco_iter_layout)

        aco_group.setLayout(aco_layout)
        layout.addWidget(aco_group)

        # 进度条
        progress_group = QGroupBox("📈 运行进度")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_label = QLabel("准备就绪")
        self.progress_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # 控制按钮（Windows 11 Fluent 风格 + 自动伸缩）
        button_group = QGroupBox("⚙️ 操作")
        button_group_layout = QVBoxLayout()
        button_group_layout.setSpacing(10)

        # 让按钮自动伸缩，不再被压缩导致文字消失
        from PyQt5.QtWidgets import QSizePolicy

        def style_button(btn):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(38)
            btn.setCursor(Qt.PointingHandCursor)

        self.btn_start = QPushButton("▶️ 开始求解")
        style_button(self.btn_start)
        self.btn_start.clicked.connect(self.start_solution)

        self.btn_stop = QPushButton("⏸️ 停止")
        style_button(self.btn_stop)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_solution)

        self.btn_reset = QPushButton("🔄 重置")
        style_button(self.btn_reset)
        self.btn_reset.clicked.connect(self.reset_solution)

        self.btn_classify = QPushButton("🤖 运行分类实验")
        style_button(self.btn_classify)
        self.btn_classify.clicked.connect(self.run_classification)

        self.btn_analysis = QPushButton("📊 参数影响分析")
        style_button(self.btn_analysis)
        self.btn_analysis.clicked.connect(self.parameter_analysis)

        self.btn_export = QPushButton("💾 导出报告")
        style_button(self.btn_export)
        self.btn_export.clicked.connect(self.export_report)

        # 添加按钮
        button_group_layout.addWidget(self.btn_start)
        button_group_layout.addWidget(self.btn_stop)
        button_group_layout.addWidget(self.btn_reset)
        button_group_layout.addWidget(self.btn_classify)
        button_group_layout.addWidget(self.btn_analysis)
        button_group_layout.addWidget(self.btn_export)

        button_group.setLayout(button_group_layout)
        layout.addWidget(button_group)

        layout.addStretch()
        return control_widget

    def create_result_panel(self):
        result_widget = QWidget()
        layout = QVBoxLayout(result_widget)
        layout.setSpacing(10)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #dcdde1;
                border-radius: 8px;
                background-color: white;
            }
        """)

        # TSP 可视化
        tsp_widget = QWidget()
        tsp_layout = QVBoxLayout(tsp_widget)
        tsp_layout.setContentsMargins(10, 10, 10, 10)

        self.fig = Figure(figsize=(14, 6), facecolor='#f5f6fa')
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.canvas = FigureCanvas(self.fig)
        tsp_layout.addWidget(self.canvas, stretch=1)

        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(220)
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                padding: 10px;
            }
        """)
        tsp_layout.addWidget(self.result_text)

        self.tab_widget.addTab(tsp_widget, "🗺️ TSP求解可视化")

        # 分类结果标签页
        class_widget = QWidget()
        class_layout = QVBoxLayout(class_widget)
        class_layout.setContentsMargins(10, 10, 10, 10)

        class_control_layout = QHBoxLayout()
        class_control_layout.addWidget(QLabel("📁 选择数据集:"))
        self.class_dataset_combo = QComboBox()
        self.class_dataset_combo.addItems(["鸢尾花 (Iris)", "葡萄酒 (Wine)", "乳腺癌 (Breast Cancer)", "汽车评估 (Car Evaluation)"])
        class_control_layout.addWidget(self.class_dataset_combo)
        class_control_layout.addStretch()
        class_layout.addLayout(class_control_layout)

        self.class_text = QTextEdit()
        self.class_text.setReadOnly(True)
        class_layout.addWidget(self.class_text)
        self.tab_widget.addTab(class_widget, "🤖 分类算法结果")

        # 参数分析标签页
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        self.tab_widget.addTab(analysis_widget, "📊 参数影响分析")

        layout.addWidget(self.tab_widget)

        self.update_plot()
        return result_widget

    def update_dataset_info(self):
        cities, optimal = TSP_DATASETS[self.current_dataset_name]
        info_text = f"城市数量: {len(cities)}"
        if optimal:
            info_text += f" | 已知最优解: {optimal:.2f}"
        else:
            info_text += " | 已知最优解: 未知"
        self.dataset_info_label.setText(info_text)

    def change_dataset(self, dataset_name):
        self.current_dataset_name = dataset_name
        self.current_cities, self.optimal_length = TSP_DATASETS[dataset_name]
        self.history = []
        self.update_dataset_info()
        self.update_plot()

    def update_plot(self, best_path=None, iteration=0, distance=0):
        self.ax1.clear()
        self.ax2.clear()

        algorithm_name = "遗传算法" if self.radio_ga.isChecked() else "蚁群算法"

        cities = self.current_cities
        x_coords = [city[0] for city in cities]
        y_coords = [city[1] for city in cities]

        self.ax1.scatter(x_coords, y_coords, c='#e74c3c', s=100, zorder=3,
                         edgecolors='white', linewidths=2)

        if len(cities) <= 30:
            for i, (x, y) in enumerate(cities):
                self.ax1.text(
                    x + 1, y + 1, str(i + 1),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )

        if best_path:
            path_x = [cities[i][0] for i in best_path]
            path_y = [cities[i][1] for i in best_path]
            path_x.append(path_x[0])
            path_y.append(path_y[0])
            self.ax1.plot(path_x, path_y, 'b-', linewidth=2.5, alpha=0.7, zorder=2)
            self.ax1.set_title(
                f'{algorithm_name} - 迭代: {iteration}, 距离: {distance:.2f}',
                fontsize=13, fontweight='bold', pad=15
            )
        else:
            self.ax1.set_title(
                f'{algorithm_name} - 城市分布图',
                fontsize=13, fontweight='bold', pad=15
            )

        self.ax1.set_xlabel('X坐标', fontsize=11, fontweight='bold')
        self.ax1.set_ylabel('Y坐标', fontsize=11, fontweight='bold')
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        self.ax1.set_facecolor('#ecf0f1')

        if self.history:
            if 'generation' in self.history[0]:
                x = [h['generation'] + 1 for h in self.history]
            else:
                x = [h['iteration'] for h in self.history]

            best_distances = [h['best_distance'] for h in self.history]
            avg_distances = [h['avg_distance'] for h in self.history]

            self.ax2.plot(x, best_distances, 'r-', label='最优解', linewidth=2.5)
            self.ax2.plot(x, avg_distances, 'b--', label='平均解', linewidth=2, alpha=0.7)
            self.ax2.set_xlabel('迭代次数', fontsize=11, fontweight='bold')
            self.ax2.set_ylabel('路径长度', fontsize=11, fontweight='bold')
            self.ax2.set_title(f'{algorithm_name} - 收敛曲线', fontsize=13, fontweight='bold', pad=15)
            self.ax2.legend(fontsize=10, framealpha=0.9)
            self.ax2.grid(True, alpha=0.3, linestyle='--')
            self.ax2.set_facecolor('#ecf0f1')
        else:
            self.ax2.text(
                0.5, 0.5,
                f'运行{algorithm_name}后显示收敛曲线\n\n点击"开始求解"按钮开始',
                ha='center', va='center', transform=self.ax2.transAxes,
                fontsize=12, color='#7f8c8d', style='italic'
            )
            self.ax2.set_title(f'{algorithm_name} - 收敛曲线', fontsize=13, fontweight='bold', pad=15)
            self.ax2.set_facecolor('#ecf0f1')

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def start_solution(self):
        if self.algorithm_thread and self.algorithm_thread.isRunning():
            return

        algorithm = "遗传算法" if self.radio_ga.isChecked() else "蚁群算法"

        try:
            if algorithm == "遗传算法":
                params = {
                    'pop_size': self.ga_pop_size.value(),
                    'elite_size': int(self.ga_pop_size.value() * 0.2),
                    'mutation_rate': self.ga_mutation.value(),
                    'generations': self.ga_generations.value()
                }
            else:
                params = {
                    'num_ants': self.aco_num_ants.value(),
                    'evaporation_rate': self.aco_evaporation.value(),
                    'alpha': 1,
                    'beta': 2,
                    'q0': 0.9,
                    'iterations': self.aco_iterations.value()
                }
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请输入有效的数值参数")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.algorithm_thread = AlgorithmThread(algorithm, self.current_cities, params)
        self.algorithm_thread.update_signal.connect(self.update_display)
        self.algorithm_thread.finished_signal.connect(self.show_final_result)
        self.algorithm_thread.progress_signal.connect(self.update_progress)
        self.algorithm_thread.start()

    def update_display(self, iteration, distance, path):
        self.update_plot(path, iteration, distance)

        optimal = self.optimal_length if self.optimal_length else distance
        error = abs(distance - optimal) / optimal * 100 if self.optimal_length else 0

        self.result_text.clear()
        self.result_text.append("=" * 60)
        self.result_text.append(f"⏱️  当前迭代: {iteration}")
        self.result_text.append(f"📏 当前最优距离: {distance:.6f}")
        if self.optimal_length:
            self.result_text.append(f"🎯 已知最优解: {optimal:.6f}")
            self.result_text.append(f"📊 相对误差: {error:.2f}%")
        self.result_text.append("=" * 60)

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def show_final_result(self, best_path, best_distance, run_time, algorithm_name):
        self.history = self.algorithm_thread.history
        self.update_plot(best_path, len(self.history), best_distance)

        optimal = self.optimal_length if self.optimal_length else best_distance

        self.result_text.clear()
        self.result_text.append("=" * 60)
        self.result_text.append("🏆 算法求解完成！")
        self.result_text.append("=" * 60)
        self.result_text.append(f"🔬 算法: {algorithm_name}")
        self.result_text.append(f"📦 数据集: {self.current_dataset_name}")
        self.result_text.append(f"🏙️  问题规模: {len(self.current_cities)}个城市")
        self.result_text.append(f"⏱️  运行时间: {run_time:.2f}秒")
        self.result_text.append(f"📏 最优距离: {best_distance:.6f}")

        if len(best_path) <= 30:
            self.result_text.append(
                f"🗺️  最优路径: {' → '.join(map(str, [i + 1 for i in best_path]))}"
            )
        else:
            self.result_text.append("🗺️  最优路径: (城市过多，已省略)")

        if self.optimal_length:
            self.result_text.append(f"\n🎯 已知最优解: {optimal:.6f}")
            self.result_text.append(f"📊 差距: {abs(best_distance - optimal):.6f}")
            self.result_text.append(
                f"📈 相对误差: {abs(best_distance - optimal) / optimal * 100:.2f}%"
            )

        self.result_text.append("\n" + "=" * 60)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText("✅ 算法完成")

    def stop_solution(self):
        if self.algorithm_thread and self.algorithm_thread.isRunning():
            self.algorithm_thread.terminate()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.progress_label.setText("⏸️ 已停止")

    def reset_solution(self):
        self.history = []
        self.update_plot()
        self.result_text.clear()
        self.result_text.append("🔄 已重置，请重新开始求解。")
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备就绪")

    def run_classification(self):
        if self.classifier_thread and self.classifier_thread.isRunning():
            return

        self.class_text.clear()
        self.class_text.append("🚀 正在运行分类实验...\n")

        dataset_map = {
            "鸢尾花 (Iris)": "iris",
            "葡萄酒 (Wine)": "wine",
            "乳腺癌 (Breast Cancer)": "breast_cancer",
            "汽车评估 (Car Evaluation)": "car"
        }
        dataset_key = dataset_map[self.class_dataset_combo.currentText()]

        self.btn_classify.setEnabled(False)
        self.progress_label.setText("🤖 正在运行分类实验...")
        self.progress_bar.setValue(0)

        self.classifier_thread = ClassifierThread(dataset_key)
        self.classifier_thread.finished_signal.connect(self.show_classification_result)
        self.classifier_thread.start()

    # ==================== 主窗口：分类结果展示（安全重建模型） ====================
    def show_classification_result(self, text, tree_info, feature_names, target_names):
        self.class_text.setPlainText(text)
        self.btn_classify.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("✅ 分类实验完成")

        # ✅ 在主线程重新训练一个决策树（仅用于可视化）
        if tree_info is not None and feature_names and target_names:
            try:
                tree_model = DecisionTreeClassifier(
                    max_depth=tree_info["max_depth"],
                    random_state=42
                )
                tree_model.fit(tree_info["X"], tree_info["y"])

                self.show_decision_tree(tree_model, feature_names, target_names)

            except Exception as e:
                QMessageBox.warning(self, "可视化失败", f"决策树绘制失败：{e}")

    # ==================== 决策树可视化 ====================
    def show_decision_tree(self, tree_model, feature_names, target_names):
        dialog = QDialog(self)
        dialog.setWindowTitle("🌲 决策树可视化")
        dialog.setGeometry(100, 100, 1200, 900)

        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(14, 10), facecolor='#f5f6fa')
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)


        plot_tree(
            tree_model,
            feature_names=feature_names,
            class_names=target_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )

        ax.set_title("决策树结构可视化", fontsize=16, fontweight='bold', pad=20)
        fig.tight_layout()

        canvas.draw()

        dialog.exec_()

    def parameter_analysis(self):
        self.analysis_text.clear()
        self.analysis_text.append("🔬 正在进行参数影响分析...\n")

        self.analysis_text.append("=" * 70)
        self.analysis_text.append("🧬 遗传算法参数影响分析")
        self.analysis_text.append("=" * 70)
        self.analysis_text.append("\n📊 1. 种群规模影响分析:")

        pop_sizes = [50, 100, 200, 300]
        results = []

        for i, pop_size in enumerate(pop_sizes):
            self.progress_bar.setValue(int((i / (len(pop_sizes) * 2)) * 100))
            self.progress_label.setText(f"🔬 分析种群规模: {i + 1}/{len(pop_sizes)}")
            QApplication.processEvents()

            start_time = time.time()
            ga = GeneticAlgorithmTSP(
                cities=cities_oliver_30[:10],
                pop_size=pop_size,
                elite_size=int(pop_size * 0.2),
                mutation_rate=0.01,
                generations=200
            )
            _, best_distance = ga.evolve()
            end_time = time.time()

            results.append({
                'pop_size': pop_size,
                'distance': best_distance,
                'time': end_time - start_time,
                'error': abs(best_distance - 166.541336) / 166.541336 * 100
            })

        self.analysis_text.append("\n┌─────────────┬──────────────┬──────────────────┬────────────────┐")
        self.analysis_text.append("│  种群规模   │  最优距离    │  运行时间(秒)    │  相对误差(%)   │")
        self.analysis_text.append("├─────────────┼──────────────┼──────────────────┼────────────────┤")
        for r in results:
            self.analysis_text.append(
                f"│  {r['pop_size']:^9}  │  {r['distance']:^10.2f}  │  {r['time']:^14.2f}  │  {r['error']:^12.2f}  │"
            )
        self.analysis_text.append("└─────────────┴──────────────┴──────────────────┴────────────────┘")

        self.analysis_text.append("\n📊 2. 变异概率影响分析:")
        mutation_rates = [0.001, 0.01, 0.05, 0.1]
        results = []

        for i, rate in enumerate(mutation_rates):
            self.progress_bar.setValue(int(((len(pop_sizes) + i) / (len(pop_sizes) * 2)) * 100))
            self.progress_label.setText(f"🔬 分析变异概率: {i + 1}/{len(mutation_rates)}")
            QApplication.processEvents()

            start_time = time.time()
            ga = GeneticAlgorithmTSP(
                cities=cities_oliver_30[:10],
                pop_size=100,
                elite_size=20,
                mutation_rate=rate,
                generations=200
            )
            _, best_distance = ga.evolve()
            end_time = time.time()

            results.append({
                'mutation_rate': rate,
                'distance': best_distance,
                'time': end_time - start_time,
                'error': abs(best_distance - 166.541336) / 166.541336 * 100
            })

        self.analysis_text.append("\n┌─────────────┬──────────────┬──────────────────┬────────────────┐")
        self.analysis_text.append("│  变异概率   │  最优距离    │  运行时间(秒)    │  相对误差(%)   │")
        self.analysis_text.append("├─────────────┼──────────────┼──────────────────┼────────────────┤")
        for r in results:
            self.analysis_text.append(
                f"│  {r['mutation_rate']:^9.3f}  │  {r['distance']:^10.2f}  │  {r['time']:^14.2f}  │  {r['error']:^12.2f}  │"
            )
        self.analysis_text.append("└─────────────┴──────────────┴──────────────────┴────────────────┘")

        self.analysis_text.append("\n" + "=" * 70)
        self.analysis_text.append("💡 分析结论")
        self.analysis_text.append("=" * 70)
        self.analysis_text.append("✅ 1. 种群规模增加会提高解的质量，但会增加计算时间")
        self.analysis_text.append("✅ 2. 变异概率需要适中，过高会导致随机性太强，过低会降低多样性")
        self.analysis_text.append("✅ 3. 参数选择需要在解质量和计算效率之间取得平衡")
        self.analysis_text.append("✅ 4. 对于10城市问题，种群规模100、变异概率0.01是较好的选择")

        self.progress_bar.setValue(100)
        self.progress_label.setText("✅ 参数分析完成")

    def export_report(self):
        report = self.generate_report()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存实验报告",
            "实验报告.txt",
            "Text files (*.txt);;All files (*.*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                QMessageBox.information(self, "导出成功", f"✅ 实验报告已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", f"保存文件时出错：{e}")

    def generate_report(self):
        report = []
        report.append("=" * 80)
        report.append("人工智能实验四：智能算法与机器学习")
        report.append("=" * 80)
        report.append("")

        if self.result_text.toPlainText().strip():
            report.append("【TSP求解结果】")
            report.append(self.result_text.toPlainText())
            report.append("")

        if self.class_text.toPlainText().strip():
            report.append("【分类算法结果】")
            report.append(self.class_text.toPlainText())
            report.append("")

        if self.analysis_text.toPlainText().strip():
            report.append("【参数影响分析】")
            report.append(self.analysis_text.toPlainText())
            report.append("")

        return "\n".join(report)


# ==================== 主程序 ====================
def main():
    app = QApplication(sys.argv)

    font = QFont()
    font.setFamily("Microsoft YaHei UI")
    font.setPointSize(9)
    app.setFont(font)

    window = AIExperimentGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
