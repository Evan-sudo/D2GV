# import os
# import numpy as np
# import torch
# import torch.optim as optim
# from entropy_model import ConditionalCodec  # 假设 Conditional Codec 保存为 entropy_model.py


# def load_npy_files(folder_path):
#     """
#     加载文件夹中的所有 .npy 文件并按名称排序。

#     Args:
#         folder_path (str): 存储 .npy 文件的文件夹路径。

#     Returns:
#         list: 按顺序排序的 .npy 文件路径列表。
#     """
#     files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
#     files.sort()  # 按文件名排序
#     return files


# def train_conditional_codec(folder_path, input_channels, quant_step, epochs=50):
#     """
#     训练 Conditional Codec 模型。

#     Args:
#         folder_path (str): 存储 .npy 文件的文件夹路径。
#         input_channels (int): 输入的通道数 C。
#         quant_step (float): 量化步长。
#         epochs (int): 训练的轮数。

#     Returns:
#         None
#     """
#     # 加载文件夹中的所有 .npy 文件
#     npy_files = load_npy_files(folder_path)
#     assert len(npy_files) > 1, "文件夹中必须至少包含两个 .npy 文件。"

#     # 初始化模型
#     model = ConditionalCodec(input_channels=input_channels, quant_step=quant_step)
#     optimizer = optim.Adam(model.parameters(), lr=1e-2)

#     # 加载第一个文件作为初始 context
#     context = torch.tensor(np.load(npy_files[0]), dtype=torch.float32)  # (N, C)

#     for epoch in range(epochs):
#         total_loss = 0.0

#         for i in range(1, len(npy_files)):  # 从第二个文件开始
#             # 加载当前文件
#             inputs = torch.tensor(np.load(npy_files[i]), dtype=torch.float32)  # (N, C)

#             # 训练模型
#             optimizer.zero_grad()
#             _, likelihood = model(inputs, context)  # context 为上一帧的输入

#             # 计算损失
#             entropy = -torch.sum(torch.log(likelihood))  # 计算熵
#             loss = entropy / inputs.shape[0]  # 平均熵作为损失

#             # 反向传播和优化
#             loss.backward()
#             optimizer.step()

#             # 累计损失
#             total_loss += loss.item()

#             # 更新 context 为当前的 inputs
#             context = inputs.clone()

#         # 打印每轮的平均损失
#         print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {total_loss / (len(npy_files) - 1):.6f}")


# if __name__ == "__main__":
#     # 参数设置
#     folder_path = "./model"  # 替换为保存 .npy 文件的文件夹路径
#     input_channels = 9  # 输入数据的通道数
#     quant_step = 0.008  # 量化步长
#     epochs = 500  # 每个文件的训练轮数

#     # 开始训练
#     train_conditional_codec(folder_path, input_channels, quant_step, epochs)


######## kd tree



# import os
# import numpy as np
# import torch
# import torch.optim as optim
# from entropy_model import ConditionalCodec  # 假设您前面实现的 Conditional Codec 保存为 conditional_codec.py
# from scipy.spatial import cKDTree


# def load_npy_files(folder_path):
#     files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
#     files.sort()  # 按文件名排序
#     return files


# def fast_compute_context_list(npy_files):
#     context_list = []

#     for i in range(len(npy_files) - 1): 
#         current_frame = np.load(npy_files[i])[:, :8] 
#         next_frame = np.load(npy_files[i + 1])[:, :8]  

#         # 构建 KD 树，用当前帧的前两个维度加速匹配
#         tree = cKDTree(current_frame)
#         _, closest_indices = tree.query(next_frame, k=1) 

#         # 存储当前帧的 context 索引
#         context_list.append(closest_indices)

#     return context_list


# def train_conditional_codec(folder_path, input_channels, quant_step, epochs=50):
#     npy_files = load_npy_files(folder_path)
#     assert len(npy_files) > 1, "文件夹中必须至少包含两个 .npy 文件。"

#     # 计算 context list
#     context_list = fast_compute_context_list(npy_files)

#     # 初始化模型
#     model = ConditionalCodec(input_channels=input_channels, quant_step=quant_step)
#     optimizer = optim.Adam(model.parameters(), lr=1e-2)

#     for epoch in range(epochs):
#         total_loss = 0.0

#         for i in range(len(context_list)):
#             current_frame = np.load(npy_files[i])  # 当前帧数据，形状为 (N, C)
#             next_frame = np.load(npy_files[i + 1])  # 下一帧数据，形状为 (N, C)

#             context_indices = context_list[i]
#             context = current_frame[context_indices]  # 使用索引构造上下文，形状为 (N, C)

#             context_tensor = torch.tensor(context, dtype=torch.float32)  # (N, C)
#             next_frame_tensor = torch.tensor(next_frame, dtype=torch.float32)  # (N, C)

#             # 模型训练
#             optimizer.zero_grad()
#             _, likelihood = model(next_frame_tensor, context_tensor)  # context 压缩 next_frame

#             # 计算损失
#             entropy = -torch.sum(torch.log(likelihood))  # 计算熵
#             loss = entropy / next_frame_tensor.shape[0]  # 平均熵作为损失

#             # 反向传播和优化
#             loss.backward()
#             optimizer.step()

#             # 累计损失
#             total_loss += loss.item()

#         # 打印每轮的平均损失
#         print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {total_loss / (len(context_list)-1):.6f}")


# if __name__ == "__main__":
#     # 参数设置
#     folder_path = "./model"  # 替换为保存 .npy 文件的文件夹路径
#     input_channels = 9  # context 的维度数
#     quant_step = 0.008  # 量化步长
#     epochs = 300  # 每个文件的训练轮数

#     # 开始训练
#     train_conditional_codec(folder_path, input_channels, quant_step, epochs)



import os
import numpy as np
import torch
import torch.optim as optim
from entropy_model import ConditionalCodec  # 假设 Conditional Codec 保存为 entropy_model.py
from scipy.spatial import cKDTree
import re



def load_npy_files(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    # 自然排序文件列表：按文件名中的数字进行排序
    files.sort(key=lambda x: [int(i) if i.isdigit() else i.lower() for i in re.split('([0-9]+)', x)])
    
    return files



def context_list_avg(npy_files, k=16):
    context_list = []

    for i in range(len(npy_files) - 1):
        # 加载当前帧和下一帧
        current_frame = np.load(npy_files[i])  # 当前帧数据，形状为 (N, C)
        next_frame = np.load(npy_files[i + 1])  # 下一帧数据，形状为 (N, C)

        # 构建 KD 树，基于前 8 维数据加速匹配
        tree = cKDTree(current_frame[:, :7])
        distances, indices = tree.query(next_frame[:, :7], k=k)

        # 计算最近 k 个点的均值作为上下文
        context_avg = np.mean(current_frame[indices], axis=1)  # 对最近 k 个点取均值
        context_list.append(context_avg)

    return context_list


def train_conditional_codec(folder_path, input_channels, quant_step, epochs=50, k=16):
    npy_files = load_npy_files(folder_path)
    print(npy_files)
    assert len(npy_files) > 1
    # 预计算 context 均值列表
    print("Calculating context averages...")
    context_list = context_list_avg(npy_files, k=k)
    print(len(context_list))
    print("Context averages calculated.")

    # 初始化模型
    model = ConditionalCodec(input_channels=input_channels, quant_step=quant_step)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(len(context_list)):
            # 加载当前帧和上下文
            next_frame = np.load(npy_files[i + 1])  # 下一帧数据，形状为 (N, C)
            context_avg = context_list[i]  # 预计算的上下文均值，形状为 (N, C)

            # 转换为 Tensor
            context_tensor = torch.tensor(context_avg, dtype=torch.float32)  # (N, C)
            next_frame_tensor = torch.tensor(next_frame, dtype=torch.float32)  # (N, C)

            # 模型训练
            optimizer.zero_grad()
            _, likelihood = model(next_frame_tensor, context_tensor)  # context 压缩 next_frame

            entropy = -torch.sum(torch.log(likelihood))  # 计算熵
            loss = entropy / next_frame_tensor.shape[0]  # 平均熵作为损失

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {total_loss / len(context_list):.6f}")


if __name__ == "__main__":
    folder_path = "./processed_model"  # 替换为保存 .npy 文件的文件夹路径
    input_channels = 7  # context 的维度数
    quant_step = 1/128  # 量化步长
    epochs = 1000  # 每个文件的训练轮数

    train_conditional_codec(folder_path, input_channels, quant_step, epochs)



