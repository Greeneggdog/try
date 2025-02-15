import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point
from datetime import datetime

# 当前时间和用户信息 (如果需要，可以保留)
CURRENT_TIME = "2025-02-10 11:58:35"
CURRENT_USER = "Greeneggdog"

class AccessibilityAnalyzer(object):
    def __init__(self):
        """医疗设施可达性分析工具"""
        self.label = "2 可达性分析"
        self.description = """
        计算医疗设施的可达性指标：
        1. 基于人口分布的空间可达性
        2. 设施服务范围分析
        3. 公平性评估
        """
        self.category = "医疗设施选址优化"
        self.canRunInBackground = False
        self.transform = None  # 初始化 transform

    def getParameterInfo(self):
        """定义工具参数"""
        # 参数0：医疗设施点数据 (Shapefile)
        param0 = {
            "name": "facility_points",
            "display_name": "医疗设施点数据",
            "type": "file",  # 使用 file 类型
            "required": True,
        }

        # 参数1：设施等级字段
        param1 = {
            "name": "level_field",
            "display_name": "设施等级字段",
            "type": "string",  # 使用 string 类型
            "required": True,
        }

        # 参数2：人口分布栅格
        param2 = {
            "name": "population_raster",
            "display_name": "人口分布栅格",
            "type": "file",
            "required": True,
        }

        # 参数3：搜索半径（米）
        param3 = {
            "name": "search_radius",
            "display_name": "搜索半径（米）",
            "type": "float",
            "default": 5000,
            "required": True,
        }

        # 参数4：距离衰减系数(β)
        param4 = {
            "name": "decay_coefficient",
            "display_name": "距离衰减系数(β)",
            "type": "float",
            "default": 1.0,
            "required": True,
        }

        # 参数5：输出文件夹
        param5 = {
            "name": "out_folder",
            "display_name": "输出文件夹",
            "type": "folder",
            "required": True,
        }
        # 参数6：上海市边界文件
        param6 = {
            "name": "shanghai_boundary",
            "display_name": "上海市边界文件(可选,用于裁剪)",
            "type": "file",
            "required": False,  # 可选
        }

        return [param0, param1, param2, param3, param4, param5, param6]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages=None):
        """执行可达性分析"""
        # 获取参数值
        facility_points_path = parameters[0]["value"]
        level_field = parameters[1]["value"]
        population_raster_path = parameters[2]["value"]
        search_radius = parameters[3]["value"]
        decay_coefficient = parameters[4]["value"]
        out_folder = parameters[5]["value"]
        shanghai_boundary_path = parameters[6]["value"]  # 可选的上海边界

        # 创建输出文件夹
        formatted_timestamp = CURRENT_TIME.replace(":", "").replace(" ", "_")
        out_workspace = os.path.join(out_folder, f"Accessibility_{formatted_timestamp}")
        if not os.path.exists(out_workspace):
            os.makedirs(out_workspace)

        # 读取设施点数据,指定编码
        facilities = gpd.read_file(facility_points_path, encoding='UTF-8') # <--- 修改这里
        # 读取人口栅格数据
        with rasterio.open(population_raster_path) as src:
            population_raster = src.read(1)
            self.transform = src.transform  # 将 transform 存储为属性
            crs = src.crs
            nodata = src.nodata

        # 统一坐标系
        if facilities.crs != crs:
            facilities = facilities.to_crs(crs)

        # 如果提供了上海边界，进行裁剪
        if shanghai_boundary_path:
            try:
                # 读取上海边界,指定编码
                shanghai = gpd.read_file(shanghai_boundary_path, encoding='UTF-8') # <--- 修改这里
                # 确保上海边界与栅格数据在同一坐标系
                if shanghai.crs != crs:
                    shanghai = shanghai.to_crs(crs)
                # 裁剪设施点
                facilities = gpd.clip(facilities, shanghai)

                # 裁剪人口栅格
                out_image, out_transform = rasterio.mask.mask(src, shanghai.geometry, crop=True, all_touched=True)
                population_raster = out_image[0]
                self.transform = out_transform # 更新transform

            except Exception as e:
                print(f"裁剪时发生错误: {e}")
                # 如果裁剪失败，可以考虑不裁剪，继续使用原始数据

        # 计算欧氏距离
        distance_surface = self.calculate_euclidean_distance(facilities, population_raster.shape, search_radius)
        # 计算可达性
        accessibility = self.calculate_gravity_accessibility(
            facilities, population_raster, distance_surface,
            decay_coefficient, level_field, nodata #移除transform
        )

        # 保存可达性栅格
        result_raster_path = os.path.join(out_workspace, "accessibility.tif")
        with rasterio.open(
            result_raster_path,
            'w',
            driver='GTiff',
            height=accessibility.shape[0],
            width=accessibility.shape[1],
            count=1,
            dtype=accessibility.dtype,
            crs=crs,
            transform=self.transform, # 使用self.transform
            nodata=nodata,
        ) as dst:
            dst.write(accessibility, 1)

        # 计算公平性指标
        equity_stats = self.calculate_equity_metrics(result_raster_path, population_raster_path)

        # 生成报告 (如果需要)
        self.generate_report(equity_stats, out_workspace)

        print("可达性分析完成！")
        return out_workspace

    def calculate_euclidean_distance(self, facilities, shape, max_distance):
        """计算欧氏距离"""
        # 创建一个空的距离栅格
        distance_raster = np.full(shape, np.inf, dtype=np.float32)

        # 将设施点转换为像素坐标
        pixel_coords = []
        for point in facilities.geometry:
            x, y = point.x, point.y
            row, col = rasterio.transform.rowcol(self.transform, x, y)  # 关键：使用 self.transform
            # 确保点在栅格范围内
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                pixel_coords.append((row, col))

        # 计算距离
        for row in range(shape[0]):
            for col in range(shape[1]):
                for facility_row, facility_col in pixel_coords:
                    dist = np.sqrt((row - facility_row)**2 + (col - facility_col)**2)
                    # 将距离转换为地理单位（例如米）
                    dist_meters = dist * self.transform.a  # 假设像元是正方形
                    if dist_meters < distance_raster[row, col]:
                        distance_raster[row, col] = dist_meters

        # 应用最大距离
        distance_raster[distance_raster > max_distance] = max_distance

        return distance_raster

    def calculate_gravity_accessibility(self, facilities, population, distance_surface, beta, level_field, nodata):
        """计算重力模型可达性"""
        # 获取设施权重 (这里简化处理，假设等级越高权重越大)
        #  如果level_field是数值类型，直接使用
        if facilities[level_field].dtype in [np.int16, np.int32, np.int64, np.float32, np.float64]:
            weights = facilities[level_field].to_numpy()
        else:
            # 如果是字符串或其他类型，需要进行映射
            # 这里提供一个示例映射，你需要根据你的实际数据进行调整
            level_mapping = {'低': 1, '中': 2, '高': 3}  # 示例映射
            weights = facilities[level_field].map(level_mapping).fillna(1).to_numpy() # 假设缺失值权重为1

        # 确保权重数量与设施数量一致
        if len(weights) != len(facilities):
            raise ValueError("权重数量与设施数量不一致")

        # 距离衰减
        decay = np.exp(-beta * distance_surface)

        # 对每个设施点应用权重
        weighted_decay_sum = np.zeros_like(decay, dtype=np.float32)
        for i, facility in facilities.iterrows():
            # 将设施点转换为像素坐标
            x, y = facility.geometry.x, facility.geometry.y
            row, col = rasterio.transform.rowcol(self.transform, x, y) #使用self.transform

            # 确保点在栅格范围内
            if 0 <= int(row) < decay.shape[0] and 0 <= int(col) < decay.shape[1]:
                # 对该设施点影响范围内的栅格累加衰减值
                weighted_decay_sum += (decay == decay[int(row), int(col)]) * weights[i]


        # 计算可达性 (与人口相乘)
        accessibility = weighted_decay_sum * population

        # 将原本无人口的区域可达性设为0
        accessibility[population <= 0] = 0

        # 设置NoData值
        if nodata is not None:
            accessibility[population == nodata] = nodata

        return accessibility

    def calculate_equity_metrics(self, accessibility_raster_path, population_raster_path):
        """计算公平性指标"""
        with rasterio.open(accessibility_raster_path) as acc_src:
            acc_array = acc_src.read(1)
            acc_nodata = acc_src.nodata

        with rasterio.open(population_raster_path) as pop_src:
            pop_array = pop_src.read(1)
            pop_nodata = pop_src.nodata

        # 过滤有效值 (同时考虑可达性和人口的NoData)
        valid_mask = (acc_array != acc_nodata) & (pop_array != pop_nodata) & (pop_array > 0) & (acc_array > 0)
        valid_acc = acc_array[valid_mask]


        if valid_acc.size == 0:
            print("没有有效的可达性值用于计算公平性指标。")
            return None

        # 计算统计量
        stats = {
            'mean_accessibility': float(np.mean(valid_acc)),
            'max_accessibility': float(np.max(valid_acc)),
            'min_accessibility': float(np.min(valid_acc)),
            'std_accessibility': float(np.std(valid_acc)),
            'valid_acc': valid_acc # 增加有效可达性值
        }

        # 计算基尼系数
        sorted_acc = np.sort(valid_acc)
        n = len(sorted_acc)
        index = np.arange(1, n + 1)
        stats['gini_coefficient'] = float(((2 * index - n - 1) * sorted_acc).sum() / (n * sorted_acc.sum()))

        # 计算变异系数
        if stats['mean_accessibility'] != 0:
            stats['cv'] = float(stats['std_accessibility'] / stats['mean_accessibility'])
        else:
            stats['cv'] = None  # 如果平均可达性为0，变异系数无意义

        return stats

    def generate_report(self, equity_stats, out_workspace):
        """生成分析报告"""
        if equity_stats is None:
            print("无法生成报告，因为缺少公平性指标。")
            return

        report_path = os.path.join(out_workspace, "accessibility_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("医疗设施可达性分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"分析时间: {CURRENT_TIME}\n")
            f.write(f"分析用户: {CURRENT_USER}\n\n")

            f.write("可达性统计：\n")
            f.write(f"平均可达性得分: {equity_stats['mean_accessibility']:.2f}\n")
            f.write(f"最大可达性得分: {equity_stats['max_accessibility']:.2f}\n")
            f.write(f"最小可达性得分: {equity_stats['min_accessibility']:.2f}\n")
            f.write(f"标准差: {equity_stats['std_accessibility']:.2f}\n\n")

            f.write("公平性指标：\n")
            f.write(f"基尼系数: {equity_stats['gini_coefficient']:.4f}\n")
            if equity_stats['cv'] is not None:
                f.write(f"变异系数: {equity_stats['cv']:.4f}\n")
            else:
                f.write("变异系数: N/A (平均可达性为0)\n")

        print(f"\n分析报告已保存至：{report_path}")