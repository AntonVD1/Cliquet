from csv_handler import CurveBook
from datetime import date


csv_path = r"C:\coding\Cliquet\dummy_curve_data.csv"
book = CurveBook(csv_path)
date_1 = date(2025/7/28)
date_2 = date(2026/7/28)
print(book)