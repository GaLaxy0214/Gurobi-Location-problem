import csv
import pandas as pd
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from gurobipy import *
## Reading shelter data
file_name = "Shelter-data.csv"
def read_shelter_coordinates(file_path):
    shelter_coordinates = []
    if file_path.endswith('.csv'):
        with open(file_path, 'r', encoding='gbk') as f:
            reader = csv.reader(f)
            # 跳过第一行（标题行）
            next(reader)
            # 读取每一行数据
            for row in reader:
                if not row:
                    continue
                # 如果该行数据有效，则提取出坐标信息
                if len(row) >= 1:
                    try:
                        didian1 = row[0]
                        longitude = float(row[1])
                        latitude = float(row[2])
                        shelter_capacity = float(row[3])  # 容量信息
                        haiba = float(row[4])
                        shelter_coordinates.append((didian1, longitude, latitude, shelter_capacity, haiba))
                    except ValueError:
                        print(f"无效的坐标信息：{row[0]},{row[1]}, {row[2]},{row[3]},{row[4]}")
    return shelter_coordinates

shelter_coordinates = read_shelter_coordinates(file_name)
# print(shelter_coordinates)

## Reading population data
file2_name = "Population-data.csv"
def read_renkou_coordinates(file2_path):
    renkou_coordinates = []
    if file2_path.endswith('.csv'):
        with open(file2_path, 'r', encoding='gbk') as f:
            reader = csv.reader(f)
            # 跳过第一行（标题行）
            next(reader)
            # 读取每一行数据
            for row in reader:
                if not row:
                    continue
                # 如果该行数据有效，则提取出坐标信息
                if len(row) >= 1:
                    try:
                        didian = row[1]
                        longitude = float(row[2])
                        latitude = float(row[3])
                        population = float(row[4])  # 人口
                        haiba1 = float(row[5])
                        renkou_coordinates.append((didian, longitude, latitude, population, haiba1))
                    except ValueError:
                        print(f"无效的坐标信息：{row[1]},{row[2]}, {row[3]},{row[4]},{row[5]},行号:{reader.line_num}")
    return renkou_coordinates

renkou_coordinates = read_renkou_coordinates(file2_name)
# print(renkou_coordinates)

# 经纬度计算距离
def distance(lon1, lat1, lon2, lat2):
    # 使用Haversine公式计算地球上两点间的距离
    R = 6371  # 地球半径（千米）
    phi1 = math.radians(lon1)
    phi2 = math.radians(lon2)
    delta_phi = math.radians(lon2 - lon1)
    delta_lambda = math.radians(lat2 - lat1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

# 计算距离矩阵
def calculate_distance_matrix(renkou_coordinates, shelter_coordinates):
    # 初始化距离矩阵
    distance_matrix = [[0.0] * len(shelter_coordinates) for _ in range(len(renkou_coordinates))]
    # 遍历两个列表，计算距离并存储到距离矩阵中
    for i in range(len(renkou_coordinates)):
        for j in range(len(shelter_coordinates)):
            d = distance(renkou_coordinates[i][1], renkou_coordinates[i][2], shelter_coordinates[j][1], shelter_coordinates[j][2])
            distance_matrix[i][j] = d
    return distance_matrix

distance_matrix = calculate_distance_matrix(renkou_coordinates, shelter_coordinates)
# print(distance_matrix)

# 将距离矩阵输出
def write_distance_matrix_to_csv(distance_matrix, renkou, shelter_coordinates, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头行
        header_row = ["", "避难所名称"] + [f"人口{i + 1}" for i in range(len(renkou))]
        writer.writerow(header_row)
        # 写入每一行数据
        for i in range(len(shelter_coordinates)):
            data_row = [f"避难所{i + 1}", shelter_coordinates[i][0]] + [f"{distance_matrix[j][i]:.2f}" for j in range(len(renkou))]
            writer.writerow(data_row)
distance_matrix = calculate_distance_matrix(renkou_coordinates, shelter_coordinates)
write_distance_matrix_to_csv(distance_matrix, renkou_coordinates, shelter_coordinates, "distance_matrix.csv")

# 在距离矩阵csv文件中写入一行新数据，人口点名称
for row in renkou_coordinates:
    renkou1 = renkou_coordinates[0]
# 打开 CSV 文件，读取现有数据
with open('distance_matrix.csv', 'r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

# 插入新行数据
new_row_data = ["人口"] + [row[0] for row in renkou_coordinates]
new_row = [""] + new_row_data + ["", ""]  # 注意要与表头行对齐

rows.insert(1, new_row)  # 在第二行下插入新行

# 将所有行写回 CSV 文件
with open('distance_matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

# distance_matrix 是一个二维数组
# shelter_coordinates 和 renkou_coordinates 是两个列表
# 其中第i个避难所对应第i列，第j个人口点对应第j行
# 将距离矩阵中小于等于10的元素设为False，其余元素设为True
# 将distance_matrix转换成NumPy数组

def find_distance_greater_than(distance_matrix):
    # 转换为numpy数组
    distance_matrix = np.array(distance_matrix)
    # 求每列的最小值
    col_min = np.min(distance_matrix, axis=0)
    # 判断哪些最小值大于10
    bool_matrix = col_min > 10
    # 获取符合条件的列的索引
    indices = np.where(bool_matrix)[0]
    # 打印结果
    if indices.size > 0:
        print("距离大于10的列索引为：", indices)
    else:
        print("所有距离均小于等于10。")
    return indices

indices = find_distance_greater_than(distance_matrix)
print(indices)



# def calculate_uncovered_points(service_areas, renkou_coordinates):
#     # 将服务范围列表转换为一个包含所有服务范围内人口点编号的集合
#     covered_points = set(point for area in service_areas for point in area)
#     # 将所有人口点编号转换为一个集合
#     all_points = set(coord[0] for coord in renkou_coordinates)
#     # 计算未被服务的人口点，即两个集合的差集
#     uncovered_points = all_points - covered_points
#     return list(uncovered_points)
# uncovered_points = calculate_uncovered_points(service_areas, renkou_coordinates)
# print("未被服务的人口点有：")
# for point in uncovered_points:
#     print(point)


# # 显示图表
# plt.show()
#######################################################
# 读取容量信息
duqushuju = read_shelter_coordinates(file_name)
shelter_capacity = []
for shelter in duqushuju:
    shelter_capacity.append(shelter[3])
# print(shelter_capacity)

# 读取人口数据
duqushuju2 = read_renkou_coordinates(file2_name)
renkou_population = []
for shelter in duqushuju2:
    renkou_population.append(shelter[3])
# print(renkou_population)

#读取海拔信息
elevations = []
for shelter in duqushuju:
    elevations.append(shelter[3])
#print(elevations)

#####################################################
# 创建模型
model = Model("binansuodiyijieduanxuanzhi")
# 添加变量
x = model.addVars(tuplelist((i, j) for i in range(len(renkou_coordinates)) for j in range(len(shelter_coordinates))), vtype=GRB.BINARY, name="x")
y = model.addVars(len(shelter_coordinates), vtype=GRB.INTEGER, lb=0, name="y")

# # 添加目标函数
# obj = quicksum(distance_matrix[i][j] * x[i, j] for i in range(len(renkou_coordinates)) for j in range(len(shelter_coordinates)))
# 所有人的疏散距离
obj = quicksum(renkou_population[i] * distance_matrix[i][j] * x[i, j] for i in range(len(renkou_coordinates)) for j in range(len(shelter_coordinates)))

model.setObjective(obj, GRB.MINIMIZE)

# 添加约束条件1：每个人口都必须分配到一个避难所，并更新避难所的剩余容量
for i in range(len(renkou_coordinates)):
    for j in range(len(shelter_coordinates)):
        model.addConstr(x[i, j] <= y[j])  # 避难所j剩余容量必须大于等于x[i, j]
    model.addConstr(quicksum(x[i, j] for j in range(len(shelter_coordinates))) == 1)  # 每个人口都必须分配到一个避难所
    for j in range(len(shelter_coordinates)):
        model.addConstr(x[i, j] <= renkou_coordinates[i][4] * y[j])  # 如果x[i, j]=1，那么y[j]减去对应的人口容量
# # 定义变量表示每个避难所的剩余容量
# capacity = [shelter_coordinates[j][4] for j in range(len(shelter_coordinates))]
# for i in range(len(renkou_coordinates)):
#     for j in range(len(shelter_coordinates)):
#         model.addConstr(x[i, j] <= y[j])
#         model.addConstr(x[i, j] <= renkou_coordinates[i][4])  # 新增约束，每个人口点的分配数不超过人口数量
#         model.addConstr(x[i, j] * renkou_coordinates[i][4] <= capacity[j])  # 新增约束，更新避难所的剩余容量
#     model.addConstr(quicksum(x[i, j] for j in range(len(shelter_coordinates))) == 1)


# # 添加约束条件1：每个人口都必须分配到一个或多个避难所，并更新避难所的剩余容量
# for i in range(len(renkou_coordinates)):
#     sub_nodes = []
#     if renkou_coordinates[i][4] > 3000:
#         sub_node1 = (i, renkou_coordinates[i][1], renkou_coordinates[i][2], renkou_coordinates[i][3], renkou_coordinates[i][4] - 3000)
#         sub_node2 = (i, renkou_coordinates[i][1], renkou_coordinates[i][2], renkou_coordinates[i][3], renkou_coordinates[i][4] - sub_node1[4])
#         sub_nodes.append(sub_node1)
#         sub_nodes.append(sub_node2)
#     else:
#         sub_nodes.append((i,) + renkou_coordinates[i][1:])
#
#     for sub_node in sub_nodes:
#         for j in range(len(shelter_coordinates)):
#             model.addConstr(x[sub_node[0], j] <= y[j])
#         model.addConstr(quicksum(x[sub_node[0], j] for j in range(len(shelter_coordinates))) == 1)
#         for j in range(len(shelter_coordinates)):
#             model.addConstr(x[sub_node[0], j] <= sub_node[4] * y[j])

# 添加约束条件2：每个避难所的容量不能超过其承载能力
for j in range(len(shelter_coordinates)):
    model.addConstr(quicksum(renkou_coordinates[i][3] * x[i, j] for i in range(len(renkou_coordinates))) <= shelter_coordinates[j][3])

# for i in range(len(renkou_coordinates)):
#     shelter_capacity1 = shelter_coordinates[i][3]
#     excess_population = []
#     for j in range(len(shelter_coordinates)):
#         if x[i, j].X > 0:
#             if shelter_capacity1 >= renkou_coordinates[i][3]:
#                 model.addConstr(x[i, j] <= y[j])
#                 shelter_capacity1 -= renkou_coordinates[i][3]
#             else:
#                 excess_population.append((i, j, renkou_coordinates[i][3] - shelter_capacity1))
#                 model.addConstr(x[i, j] == 0)
#     if shelter_capacity1 < 0:
#         print("Error: Shelter capacity is negative.")
#         break
#
# if excess_population:
#     print("The following population coordinates have excess population:")
#     for item in excess_population:
#         print(f"Population at ({renkou_coordinates[item[0]][1]}, {renkou_coordinates[item[0]][2]}) has excess population of {item[2]} at shelter {item[1]}.")
# else:
#     print("All population has been allocated to shelters successfully.")


# 添加约束条件3：每个选址变量必须为0或1
for i in range(len(renkou_coordinates)):
    for j in range(len(shelter_coordinates)):
        model.addConstr(x[i, j] >= 0)
        model.addConstr(x[i, j] <= 1)
# 添加约束条件4：每个人口点被分配的避难所距离不能超过10
for i in range(len(renkou_coordinates)):
    for j in range(len(shelter_coordinates)):
        model.addConstr((distance_matrix[i][j] * x[i, j]) <= 10)
# 添加约束条件5：每个避难所接收的居民数等于该避难所对应的选址变量所分配的居民数之和
for j in range(len(shelter_coordinates)):
    model.addConstr(quicksum(renkou_coordinates[i][3] * x[i, j] for i in range(len(renkou_coordinates))) == y[j])
# 约束6：避难所容量松弛约束
for j in range(len(shelter_capacity)):
    model.addConstr(quicksum(x[i, j] * renkou_population[i] for i in range(len(renkou_population))) <= shelter_capacity[j] * 1.2)

# 添加约束条件:7：每个避难所都必须至少分配一个人口
for j in range(len(shelter_coordinates)):
    model.addConstr(quicksum(x[i, j] for i in range(len(renkou_coordinates))) >= 1)

# 添加约束条件8：避难所的容量必须是一个非负整数
for j in range(len(shelter_coordinates)):
    model.addConstr(y[j] >= 0)

# 添加约束条件9：如果某个避难所没有被选中，那么它对应的变量y_j必须为0
for j in range(len(shelter_coordinates)):
    model.addConstr(y[j] >= quicksum(x[i, j] for i in range(len(renkou_coordinates))) - 1)



# 设置输出日志文件名
model.setParam(GRB.Param.LogFile, "logFile.log")
model.write("constraints.lp")
for j in range(len(shelter_coordinates)):
    y[j].start = shelter_coordinates[j][3]  # 初始值为避难所的承载能力

# 求解模型
model.Params.TimeLimit = 60 # 设置时间限制为1小时
model.optimize()

# # 直接结果输出
# if model.status == GRB.OPTIMAL:
#     print("最短距离为：", model.objVal)
#     for i in range(len(renkou_coordinates)):
#         for j in range(len(shelter_coordinates)):
#             if x[i, j].x == 1:
#                 print("人口", i, "被分配到避难所", j)
# else:
#     print("模型求解失败")

# 输出所有居民点的疏散距离和目标
output_file = "所有居民点的疏散距离.xlsx"  # 输出文件的名称
data = []  # 存储结果的列表

if model.status == GRB.OPTIMAL:
    print("最短距离为：", model.objVal)
    for i in range(len(renkou_coordinates)):
        for j in range(len(shelter_coordinates)):
            if x[i, j].x == 1:
                evacuation_distance = distance_matrix[i][j]  # 计算疏散距离
                renkou_name = renkou_coordinates[i][0]  # 人口点 i 的名称
                shelter_name = shelter_coordinates[j][0]  # 避难所 j 的名称
                data.append([renkou_name, shelter_name, evacuation_distance])

    df = pd.DataFrame(data, columns=["人口点", "避难所", "疏散距离"])  # 创建 DataFrame 对象
    df.to_excel(output_file, index=False)  # 将 DataFrame 写入 Excel 文件
    print("结果已成功输出到文件:", output_file)
else:
    print("模型求解失败")

# # 检查x值
# for i in range(len(renkou_coordinates)):
#     for j in range(len(shelter_coordinates)):
#         if x[i,j].x == 1:
#             print("x[{},{}] = {}".format(i,j,x[i,j].x))


# def save_allocation(renkou_coordinates, shelter_coordinates, x):
#     # 创建一个Workbook对象
#     wb = Workbook()
#
#     # 创建一个名为"Allocation"的工作表
#     ws = wb.create_sheet("Allocation")
#
#     # 在第一行写入表头
#     ws.append(["Renkou ID", "Selected Shelter ID"])
#
#     # 遍历每个人口和避难所，如果它们之间的分配变量x为1，则将此分配记录写入工作表
#     for i in range(len(renkou_coordinates)):
#         for j in range(len(shelter_coordinates)):
#             if x[i,j].x == 1:
#                 ws.append([renkou_coordinates[i][0], shelter_coordinates[j][0]])
#     # 保存Workbook对象到文件中
#     wb.save("Allocation.xlsx")
# save_allocation(renkou_coordinates, shelter_coordinates, x)

# 创建一个字典来保存居民点的分配方案
allocations = {}
# 遍历所有居民点，确定其分配方案
for i in range(len(renkou_population)):
    for j in range(len(shelter_capacity)):
        if x[(i,j)].X == 1.0:
            distance = distance_matrix[i][j]
            allocations[i] = {'shelter': j, 'ratio': x[(i,j)].X, 'distance': distance}

# 遍历字典，输出每个居民点的分配方案
for i in allocations:
    shelter = allocations[i]['shelter'] + 1  # 避难所编号加 1，从 1 开始计数
    ratio = allocations[i]['ratio']
    distance = allocations[i]['distance']
    population = renkou_population[i]
    allocated_population = round(population * ratio)  # 四舍五入保留整数
    print(f"居民点 {i+1} 的分配方案为：分配到避难所 {shelter}，分配比例为 {ratio}，分配人数为 {allocated_population}，疏散距离为 {distance}")



# 统计每个避难所收容的人口点
allocations1 = {}
shelter_allocations = {}
# 遍历所有居民点，确定其分配方案
for i in range(len(renkou_population)):
    for j in range(len(shelter_capacity)):
        if x[(i,j)].X == 1.0:
            allocations1[i] = {'shelter': j, 'ratio': x[(i,j)].X}
            if j not in shelter_allocations:
                shelter_allocations[j] = []
            shelter_allocations[j].append(i)

# 输出每个避难所收纳的人口点
for j in shelter_allocations:
    print(f"避难所 {j+1} 收纳了以下居民点：{shelter_allocations[j]}")

# 统计每个避难所的容纳人数
shelter_populations = [0] * len(shelter_capacity)  # 初始化为 0
for i in allocations:
    shelter = allocations[i]['shelter']
    ratio = allocations[i]['ratio']
    population = renkou_population[i]
    allocated_population = round(population * ratio)  # 四舍五入保留整数
    shelter_populations[shelter] += allocated_population  # 累加容纳人数

# 输出每个避难所的容纳人数和总容纳人数
total_population = sum(shelter_populations)
# 输出每个避难所的人数
# for i, population in enumerate(shelter_populations):
#     print(f"避难所 {i+1} 的容纳人数为：{population}")
print(f"总的容纳人数为：{total_population}")

# 输出结果表格
def output_results(model, shelter_populations, total_population):
    # 创建一个空的列表
    table_data = []
    # 添加表头
    table_data.append(['避难所编号', '容纳人数'])
    # 遍历每个避难所，将编号和容纳人数添加到列表中
    for i, population in enumerate(shelter_populations):
        table_data.append([i+1, population])
    # 将总容纳人数添加到列表中
    table_data.append(['总容纳人数', total_population])
    # 将列表写入到CSV文件中
    with open('shelter_population.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(table_data)
    print('表格文件已生成！')
    # 输出总疏散距离
    print("Total distance: ", model.getObjective().getValue())
output_results(model, shelter_populations, total_population)

# 在模型求解后遍历所有避难所，获取避难所剩余容量变量的值
for j in range(len(shelter_coordinates)):
    # 获取避难所剩余容量变量的值
    shelter_capacity1 = y[j].getAttr('X')

    # 遍历所有已分配到该避难所的人口，减去其对应的人口容量
    for i in range(len(renkou_coordinates)):
        if x[(i, j)].X == 1.0:
            shelter_capacity1 -= renkou_coordinates[i][4]

    # print("避难所 {} 的剩余容量为: {}".format(shelter_coordinates[j][0], shelter_capacity1))

# # 打印所有x为1的值
# for i in range(len(renkou_coordinates)):
#     for j in range(len(shelter_coordinates)):
#         if x[i, j].X > 0:
#             print(f"x[{i},{j}] = {x[i,j].X}")

# # 打印y
# for j in range(len(shelter_coordinates)):
#     print(f"Shelter {j+1} capacity: {y[j].X}")

#################################################################
# 绘图
# 绘制疏散图
plt.rcParams['font.family'] = 'SimHei'
fig = plt.figure(figsize=(8, 6))
fig.suptitle('避难所选址疏散图')
ax = fig.add_subplot(111)
ax.set_title('地图')
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
new_dict = {}
for key, value in allocations.items():
    shelter_index = value['shelter']
    new_dict[key] = shelter_index
# print(new_dict)


# 添加标题和标签
plt.title('算法迭代')
plt.xlabel('迭代次数')
plt.ylabel('总疏散距离')
# 绘制人口点和避难所的散点图
renkou_x = [coord[1] for coord in renkou_coordinates]
renkou_y = [coord[2] for coord in renkou_coordinates]
shelter_x = [coord[1] for coord in shelter_coordinates]
shelter_y = [coord[2] for coord in shelter_coordinates]
plt.scatter(renkou_x, renkou_y, color='blue')
plt.scatter(shelter_x, shelter_y, color='red')

# 绘制连线
for i, shelter_index in new_dict.items():
    plt.plot([renkou_coordinates[i][1], shelter_coordinates[shelter_index][1]],
             [renkou_coordinates[i][2], shelter_coordinates[shelter_index][2]], color='black')
# 显示图像
plt.show()