# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:38:02 2022

@author: Li Hangfeng
"""

import math


class EuclideanDistTracker:
    def __init__(self):
        # The position that store the central point location
        self.center_points = {}
        
        # The count is incremented by one each time a new object ID is detected
        self.id_count = 0


    def update(self, objects_rect):
        # Object box and ID
        objects_bbs_ids = []

        # Gets the center point of the new object
        for rect in objects_rect:
            x, y = rect
            cx = x
            cy = y

            # Find out if the object has been detected
            same_object_detected = False
            for id, pt in self.center_points.items():
            	# Compute the Euclidean distance between the center points
                dist = math.hypot(cx - pt[0], cy - pt[1])
				# If the Euclidean distance is less than 10, it indicates the same target
                if dist < 10:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, id])
                    same_object_detected = True
                    break

            # A new object is detected and we assign an ID to the object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, self.id_count])
                self.id_count += 1

        # The dictionary is cleaned by the center point to remove IDS that are no longer used
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _,object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update the dictionary
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
