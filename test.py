import pandas as pd

# Đường dẫn tới tệp Excel
input_file = 'C:\DEANTN\genarate_question\datafn.xls'  # hoặc .xlsx
output_file = 'C:\DEANTN\genarate_question\data.csv'

# Đọc tệp Excel
data = pd.read_excel(input_file)

# Ghi tệp CSV
data.to_csv(output_file, index=False, encoding='utf-8')

print(f"Đã chuyển đổi thành công {input_file} sang {output_file}")
