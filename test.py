import pandas as pd
import re

# Đường dẫn tới tệp Excel
input_file = r'/workspaces/exam-generator/datafn.xls'  # Use raw string for paths
output_file = r'/workspaces/exam-generator/data.csv'

# Đọc tệp Excel
data = pd.read_excel(input_file)

# Làm sạch cột câu hỏi: Xóa tiền tố như "Câu 1" hoặc "Câu 10:" với hoặc không có dấu ":"
data['CAUHOI'] = data['CAUHOI'].apply(lambda x: re.sub(r"^(Câu|câu|cau|Cau)\s+\d+\s*:? ?", "", x))

# Làm sạch cột câu trả lời: Xóa lặp kiểu "A. A."
data['CAUTRALOI'] = data['CAUTRALOI'].apply(lambda x: '~'.join(
    re.sub(r"^[A-Z]\.\s*[A-Z]\.\s*", "", answer.strip()) for answer in x.split('~')
))

# Ghi tệp CSV
data.to_csv(output_file, index=False, encoding='utf-8', sep=";")

print(f"Đã chuyển đổi thành công {input_file} sang {output_file}")
