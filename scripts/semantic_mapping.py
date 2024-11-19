import database_exporter_py
from sensor_msgs.msg import PointCloud2
import cv2 as cv

exporter = database_exporter_py.DatabaseExporter('lab-20241112.db', '')
result = exporter.load_rtabmap_db()
print(f"Loaded {len(result.images)} images, {len(result.depths)} depths and {len(result.pointcloud)} points")
