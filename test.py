import pandas as pd
import re
# Đường dẫn tới tệp Excel
input_file = 'C:\DEANTN\genarate_question\datafn.xls'  # hoặc .xlsx
output_file = 'C:\DEANTN\genarate_question\data.csv'

# Đọc tệp Excel
data = pd.read_excel(input_file)



# Xóa chuỗi lặp "Câu X: Câu Y"
data['CAUHOI'] = data['CAUHOI'].apply(lambda x: re.sub(r"(Câu\s+\d+:)\s*(Câu\s+\d+:)?", "Câu ", x))

# Làm sạch đáp án: Xóa lặp kiểu "A. A."
data['CAUTRALOI'] = data['CAUTRALOI'].apply(lambda x: '~'.join(
    re.sub(r"^[A-Z]\.\s*[A-Z]\.\s*", "", answer.strip()) for answer in x.split('~')
))
# Ghi tệp CSV
data.to_csv(output_file, index=False, encoding='utf-8', sep=";")

print(f"Đã chuyển đổi thành công {input_file} sang {output_file}")
