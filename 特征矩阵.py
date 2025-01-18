import geopandas as gpd
from tkinter import Tk
import numpy as np
from tkinter.filedialog import askopenfilename
from scipy.sparse import csr_matrix, save_npz  # 用于创建和保存稀疏矩阵


def read_shapefile(file_path):
    """读取Shapefile并验证面要素类型"""
    gdf = gpd.read_file(file_path)
    if not all(gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])):
        raise ValueError("该文件不包含面要素，请提供包含面要素的文件。")
    return gdf


def create_relation_matrix(gdf, fid_column):
    """基于面要素的几何关系创建一个合并后的关系矩阵"""
    fid_to_index = {fid: idx for idx, fid in enumerate(gdf[fid_column])}
    num_nodes = len(gdf)

    # 初始化关系矩阵，默认全部为 0（无连接）
    relation_matrix = np.zeros((num_nodes, num_nodes))

    # 填充关系矩阵
    for idx1, poly1 in gdf.iterrows():
        for idx2, poly2 in gdf.iterrows():
            if idx1 >= idx2:
                continue

            # 相交关系 (相交为 1)
            if poly1.geometry.intersects(poly2.geometry):
                relation_matrix[fid_to_index[poly1[fid_column]], fid_to_index[poly2[fid_column]]] = 1
                relation_matrix[fid_to_index[poly2[fid_column]], fid_to_index[poly1[fid_column]]] = 1  # 矩阵对称

            # 包含关系 (包含为 2)
            if poly1.geometry.contains(poly2.geometry):
                relation_matrix[fid_to_index[poly1[fid_column]], fid_to_index[poly2[fid_column]]] = 2
            if poly2.geometry.contains(poly1.geometry):
                relation_matrix[fid_to_index[poly2[fid_column]], fid_to_index[poly1[fid_column]]] = 2

            # 可以根据需要加入其他关系，比如相离等
            # 例如：如果两个面没有任何关系，可以标为 0

    return relation_matrix, fid_to_index


def save_relation_matrix_with_mapping(matrix, mapping, filename_matrix, filename_mapping):
    """将关系矩阵和编号与value的映射保存"""
    sparse_matrix = csr_matrix(matrix)
    save_npz(filename_matrix, sparse_matrix)  # 保存稀疏矩阵

    # 保存编号与 `value` 值的映射
    with open(filename_mapping, 'w') as f:
        f.write("Index\tValue\n")
        for value, index in mapping.items():
            f.write(f"{index}\t{value}\n")

    print(f"关系矩阵已保存为稀疏格式：{filename_matrix}")
    print(f"编号与 value 的映射已保存为：{filename_mapping}")


# 主程序
if __name__ == "__main__":
    # 打开文件选择对话框
    Tk().withdraw()  # 隐藏主窗口
    file_path = askopenfilename(title="选择 Shapefile 文件",
                                filetypes=[("Shapefile", "*.shp"), ("GeoPackage", "*.gpkg")])

    if not file_path:
        print("没有选择文件，程序退出。")
        exit()

    try:
        gdf = read_shapefile(file_path)
        print("Shapefile 列名:", gdf.columns)

        # 假设 'value' 列是唯一标识符（修改为您的实际列名）
        fid_column = 'value'
        if fid_column not in gdf.columns:
            raise ValueError(f"列名 '{fid_column}' 不存在于数据中。")

        # 计算关系矩阵
        relation_matrix, fid_to_index = create_relation_matrix(gdf, fid_column)

        # 保存关系矩阵和映射关系
        save_relation_matrix_with_mapping(
            relation_matrix,
            fid_to_index,
            'relation_matrix.npz',
            'index_to_value_mapping.txt'
        )

    except Exception as e:
        print(f"发生错误: {e}")
