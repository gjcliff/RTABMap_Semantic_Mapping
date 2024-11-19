import database_exporter_py

exporter = database_exporter_py.DatabaseExporter('lab-20241112.db', '')
exporter.load_rtabmap_db()
