# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:38:02 2022

@author: e0947330
"""

import math


class EuclideanDistTracker:
    def __init__(self):
        # 存储对象的中心位置
        self.center_points = {}
        # 保持id的计数
        # 每次检测到新的对象 id 时，计数将增加 1
        self.id_count = 0


    def update(self, objects_rect):
        # 对象框和 ID
        objects_bbs_ids = []

        # 获取新对象的中心点
        for rect in objects_rect:
            x, y = rect
            cx = x
            cy = y

            # 查明是否已经检测到该对象
            same_object_detected = False
            for id, pt in self.center_points.items():
            	# 计算中心点之间的欧式距离
                dist = math.hypot(cx - pt[0], cy - pt[1])
				# 如果欧氏距离小于25即表明是相同的目标
                if dist < 10:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, id])
                    same_object_detected = True
                    break

            # 检测到新对象，我们将 ID 分配给该对象
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, self.id_count])
                self.id_count += 1

        # 按中心点清理字典以删除不再使用的 IDS
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _,object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # 更新字典
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
