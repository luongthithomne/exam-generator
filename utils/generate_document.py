import os
from typing import List
from fpdf import FPDF
from model.question import Question


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(10, 10, 10)  # Set margins: left, top, right
        self.set_auto_page_break(auto=True, margin=15)  # Auto page break with a margin of 15 at the bottom
        self.first_page = True  # Biến để theo dõi trang đầu tiên

    def header(self):
        if self.first_page:  # Chỉ in header ở trang đầu tiên
            self.add_font('DejaVuSans', '', 'dejavu-sans/DejaVuSans.ttf', uni=True)
            self.add_font('DejaVuSans', 'B', 'dejavu-sans/DejaVuSans-Bold.ttf', uni=True)
            
            # Định dạng hàng đầu tiên: Căn trái và căn phải
            self.set_font('DejaVuSans', '', 12)
            self.cell(100, 10, 'SỞ GIÁO DỤC VÀ ĐÀO TẠO', 0, 0, 'L')
            self.cell(90, 10, 'ĐỀ KIỂM TRA CUỐI HỌC KỲ I', 0, 1, 'R')
            
            # Định dạng hàng thứ hai
            self.cell(100, 10, 'THÀNH PHỐ HỒ CHÍ MINH', 0, 0, 'L')
            self.cell(90, 10, 'MÔN: TIN HỌC - KHỐI 10', 0, 1, 'R')
            
            # Định dạng hàng thứ ba
            self.cell(100, 10, 'TRƯỜNG THPT BÀ ĐIỂM', 0, 0, 'L')
            self.cell(90, 10, 'Thời gian làm bài: 45 phút', 0, 1, 'R')

            
            self.cell(0, 10, 'Họ và tên học sinh: ..................', 0, 1, 'L')
            self.cell(0, 10, 'SBD: ..................', 0, 1, 'L')
            self.cell(0, 10, 'Lớp: 11 ..................', 0, 1, 'L')
            self.cell(0, 10, 'Mã đề: 304', 0, 1, 'L')
            self.cell(0, 10, '', 0, 1)  # Dòng trống
            self.first_page = False  # Đánh dấu là không còn là trang đầu tiên

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVuSans', '', 8)
        self.cell(0, 10, f'Trang {self.page_no()}', 0, 0, 'C')


# def create_pdf(questions: List[Question], filename: str):
#     pdf = PDF()
#     pdf.add_page()
#
#     # Thêm font vào PDF
#     pdf.add_font('DejaVuSans', '', 'dejavu-sans/DejaVuSans.ttf', uni=True)
#     pdf.set_font('DejaVuSans', '', 12)
#
#     for question in questions:
#         # In câu hỏi với định dạng in đậm
#         pdf.set_font('DejaVuSans', 'B', 12)  # Đặt font in đậm
#         pdf.cell(0, 10, f"Câu {question.id}: {question.question}", 0, 1)
#
#         pdf.set_font('DejaVuSans', '', 12)  # Quay lại font thường
#         for index, answer in enumerate(question.answers):
#             # Đánh số đáp án từ A đến F
#             pdf.cell(0, 10, f"{chr(65 + index)}. {answer}", 0, 1)  # 65 là mã ASCII của 'A'
#
#         pdf.cell(0, 10, "", 0, 1)  # Dòng trống giữa các câu hỏi
#
#     pdf.cell(0, 10, "---------- HẾT ----------", 0, 1, 'C')
#
#     pdf.output(filename, 'F')  # Lưu PDF

# def create_pdf(questions: List[Question], filename: str):
#     pdf = PDF()
#     pdf.add_page()
#     pdf.add_font('DejaVuSans', '', 'dejavu-sans/DejaVuSans.ttf', uni=True)
#     pdf.set_font('DejaVuSans', '', 12)
#
#     for question in questions:
#         pdf.set_font('DejaVuSans', 'B', 12)
#         pdf.multi_cell(190, 10, f"Câu {question.id}: {question.question}", border=0)
#
#         pdf.set_font('DejaVuSans', '', 12)
#         for index, answer in enumerate(question.answers):
#             pdf.multi_cell(190, 10, f"{chr(65 + index)}. {answer}", border=0)
#
#         pdf.cell(190, 10, "", 0, 1)  # Empty line between questions
#
#     pdf.cell(190, 10, "---------- HẾT ----------", 0, 1, 'C')
#     pdf.output(filename, 'F')  # Lưu PDF

def create_pdf(questions: List[Question], filename: str):
    pdf = PDF()
    pdf.add_page()
    pdf.add_font('DejaVuSans', '', 'dejavu-sans/DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVuSans', '', 12)

    for question in questions:
        pdf.set_font('DejaVuSans', 'B', 12)
        pdf.multi_cell(190, 10, f"Câu {question.id}: {question.question}", 0, 'L')
        y_after_question = pdf.get_y()  # Lấy vị trí y hiện tại sau khi in câu hỏi

        pdf.set_font('DejaVuSans', '', 12)
        for index, answer in enumerate(question.answers):
            pdf.set_xy(10, y_after_question)  # Đặt lại vị trí x bắt đầu in câu trả lời
            pdf.multi_cell(190, 10, f"{chr(65 + index)}. {answer}", 0, 'L')
            y_after_question = pdf.get_y()  # Cập nhật vị trí y sau mỗi câu trả lời

        pdf.ln(10)  # Thêm khoảng cách dòng sau mỗi nhóm câu hỏi và câu trả lời

    pdf.cell(0, 10, "---------- HẾT ----------", 0, 1, 'C')
    pdf.output(filename, 'F')  # Save the PDF

def questions_to_pdf(questions: List[Question], output_file: str):
    create_pdf(questions, output_file)
