import json
from datetime import datetime
import os
from abc import abstractmethod
from typing import Optional
from model.question import Question
from utils.api import get_questions, clarify_question, get_questions_from_bank
from utils.generate_document import questions_to_pdf
import pandas as pd
import streamlit as st
import re

ROOT_PATH = '/mount/src/exam-generator'
# ROOT_PATH = '/Users/taihv/Documents/1.Primary/Thom/Final/gitv2/exam-gen1erator'
class PageEnum:
    """
    Enum for pages
    """
    GENERATE_EXAM = 0
    QUESTIONS = 1
    RESULTS = 2

logo_path = os.path.join(os.path.dirname(__file__), '..', 'media', 'logo.png')

class Page:

    @abstractmethod
    def render(self, app):
        """
        Render the page (must be implemented by subclasses)
        """

import json
import streamlit as st
import pandas as pd

class GenerateExamPage(Page):

    description = """
    <p>Chào mừng bạn đến với hệ thống tạo đề trắc nghiệm thông minh, một công cụ hỗ trợ giáo viên trong việc tạo đề thi một cách nhanh chóng!</p>

    <p class="text_title">Tính năng nổi bật của hệ thống:</p>
    <ul>
        <li>Hỗ trợ giáo viên tạo nhanh các đề thi trắc nghiệm phù hợp với ma trận đề thi</li>
        <li>Tự động tạo ra câu hỏi mới đáp ứng ma trận đề tích hợp API thông minh</li>
        <li>Tạo đề thi không trùng lặp với các đề thi đã tạo trước đó</li>
    </ul>
    """

    def render(self, app):
        st.markdown(
            """
            <style>
            .main {
                background-color: #f7f9fc;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                border: 2px solid #4A90E2;
            }
            h1 {
                text-align: center;
                font-size: 2.5em;
                font-weight: bold;
                color: #4A90E2;
                margin-bottom: 10px;
            }
            h2 {
                text-align: center;
                color: #555555;
                margin-bottom: 20px;
            }
            p {
                font-size: 1.1em;
                line-height: 1.6;
                margin: 15px 0;
            }
            .logo {
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100px;
                height: 100px;
            }
            .text_title{
                font-weight: 800;
            }
            </style>
            """, unsafe_allow_html=True
        )

        # Replace 'path/to/logo.png' with your actual logo path or URL
        st.image(ROOT_PATH + "/app/logo.png", use_column_width=True)  # Automatically centers the image based on container width
        st.markdown("<h1>Hệ thống thông minh tạo đề kiểm tra trắc nghiệm môn Tin học 10</h1>", unsafe_allow_html=True)
        st.markdown(self.description, unsafe_allow_html=True)


        # Load CSV data
        csv_file_path = ROOT_PATH + '/data.csv'
        data = pd.read_csv(csv_file_path, delimiter=';', on_bad_lines='skip')
        data = data[['SACH', 'BAI', 'CHUDE', 'MUCDO', 'NOIDUNG_YCCD']].dropna()

        # Initialize session state for selected items if not already done
        if 'selected_bai_info' not in st.session_state:
            st.session_state.selected_bai_info = []
        if 'questions_per_mucdo' not in st.session_state:
            st.session_state.questions_per_mucdo = {mucdo: 0 for mucdo in data['MUCDO'].unique()}

        # Initialize session state for selections
        if 'selected_sach' not in st.session_state:
            st.session_state.selected_sach = []
        if 'selected_bai' not in st.session_state:
            st.session_state.selected_bai = None
        if 'selected_chude' not in st.session_state:
            st.session_state.selected_chude = []
        if 'selected_yccd' not in st.session_state:
            st.session_state.selected_yccd = []

        # Create combo box to select multiple 'Sách'
        #selected_sach = st.multiselect("Chọn Sách", data['SACH'].unique(), key="sach_select", default=st.session_state.selected_sach)
        selected_sach = st.selectbox("Chọn Sách", data['SACH'].unique(), key="sach_select")
        # Logic for selecting bài
        if selected_sach:

             # Choose 'Chủ Đề' (single choice)
            chude_options = data[data['SACH'] == selected_sach]['CHUDE'].unique()
            selected_chude = st.selectbox("Chọn Chủ Đề", chude_options, key="chude_select")

            # # Choose Chủ Đề (multiple choice)
            # chude_options = data[data['SACH'].isin(selected_sach)]['CHUDE'].unique()  #data[data['BAI'] == selected_bai]['CHUDE'].unique()
            # selected_chude = st.multiselect("Chọn Chủ Đề", chude_options, key="chude_select", default=st.session_state.selected_chude)
            
            if selected_chude:
                # Create a select box to choose a single 'Bài'
                bai_options = data[data['CHUDE'] == selected_chude]['BAI'].unique()
                selected_bai = st.selectbox("Chọn Bài", bai_options, key="bai_select")

                
                # Choose Yêu Cầu Cần Đạt (multiple choice)
                yccd_options = data[data['BAI'] == selected_bai]['NOIDUNG_YCCD'].unique()
                # Làm sạch các chuỗi trong yccd_options bằng cách loại bỏ dấu ~ ở cuối
                cleaned_yccd_options = [re.sub(r"~+$", "", yccd) for yccd in yccd_options]

                # Sử dụng giá trị đã làm sạch cho multiselect
                selected_yccd = st.multiselect(
                    "Chọn Yêu Cầu Cần Đạt",
                    cleaned_yccd_options,
                    key="yccd_select",
                    default=st.session_state.selected_yccd
                )
                # Input number of questions for each level
                st.header("Nhập số lượng câu hỏi cho mức độ")
                for mucdo in st.session_state.questions_per_mucdo.keys():
                    st.session_state.questions_per_mucdo[mucdo] = st.number_input(
                        f"Số lượng câu hỏi cho mức độ '{mucdo}'",
                        min_value=0,
                        value=st.session_state.questions_per_mucdo[mucdo],
                        step=1,
                        key=f"questions_{mucdo}"
                    )

            # Option to continue selecting more bài
            if st.button("Thực hiện"):
                # Clear current selections
                st.session_state.selected_bai_info.append({
                    "Sách": selected_sach,
                    "Bài": selected_bai,
                    "Chủ Đề": selected_chude,
                    "Yêu Cầu Cần Đạt": selected_yccd,
                    "Số lượng câu hỏi": st.session_state.questions_per_mucdo.copy()  # Store a copy of the current question counts
                })

                # Reset selections for new input
                st.session_state.selected_bai = None  # Reset bài selection
                st.session_state.selected_chude = []  # Reset Chủ Đề
                st.session_state.selected_yccd = []  # Reset Yêu Cầu Cần Đạt
                st.session_state.selected_sach = selected_sach  # Keep the selected sách
                for mucdo in st.session_state.questions_per_mucdo.keys():
                    st.session_state.questions_per_mucdo[mucdo] = 0  # Reset question counts

                st.rerun()  # Reload the page to allow new selections
            
                # Clear session state for the next selection
            def clear_session_state():
                for key in st.session_state.keys():
                    del st.session_state[key]

            # Option to continue selecting more bài
            if st.button("Chọn lại"):
                clear_session_state()  # Call the function to clear session state
                st.rerun()  # Reload the page to allow new selections

            # Display saved bài information
            # if st.session_state.selected_bai_info:
            #     st.write("Thông tin các bài đã chọn:")
            #     for info in st.session_state.selected_bai_info:
            #         st.write(info)

            

            # Tạo DataFrame từ thông tin đã chọn
            selected_info = st.session_state.selected_bai_info
            flattened_data = []

            for info in selected_info:
                for mucdo, count in info['Số lượng câu hỏi'].items():
                    flattened_data.append({
                        "Sách": info["Sách"],
                        "Bài": info["Bài"],
                        "Chủ Đề": info["Chủ Đề"],
                        "Yêu Cầu Cần Đạt": '\n\n'.join(info["Yêu Cầu Cần Đạt"]),
                        "Mức Độ": mucdo,
                        "SL": count if count is not None else 0
                    })

            # Chuyển dữ liệu thành DataFrame
            df = pd.DataFrame(flattened_data)

            if(not df.empty):
                # Gộp các hàng có cùng Sách, Bài, Chủ Đề, Yêu Cầu Cần Đạt
                grouped = df.pivot_table(
                    index=["Sách", "Bài", "Chủ Đề", "Yêu Cầu Cần Đạt"],
                    columns="Mức Độ",
                    values="SL",
                    aggfunc="sum",
                    fill_value=0,  # Thay giá trị NaN bằng 0
                ).reset_index()

                # Đổi tên các cột để hiển thị rõ ràng
                grouped.columns.name = None  # Xóa tên của cột
                grouped = grouped.rename(columns={"Thông hiểu": "TH", "Nhận biết": "NB", "Vận dụng": "VD"})

                # Hiển thị dưới dạng bảng
                st.table(grouped)


        # Calculate the total number of questions based on user input
        total_questions = sum(sum(info['Số lượng câu hỏi'].values()) for info in st.session_state.selected_bai_info)
        st.markdown(f"### Tổng số câu hỏi: {total_questions}")

        # number_of_answers = st.number_input(
        #     "Số lượng đáp án",
        #     min_value=3,
        #     max_value=5,
        #     value=4,
        #     help=f"Tổng số câu hỏi: {total_questions}"
        # )
        number_of_answers = 4

        if st.button("Tạo đề thi từ ngân hàng câu hỏi", help="Generate the questions according to the parameters"):
            st.warning("Hệ thống đang tạo câu hỏi. Bạn vui lòng chờ trong giây lát...")

            # Prepare dictionary for all selected information
            print('---------------------------------------------------------------- start')
            exam_params = {
                "Sách": [info["Sách"] for info in st.session_state.selected_bai_info],
                "Bài": [info["Bài"] for info in st.session_state.selected_bai_info],
                "Chủ Đề": [info["Chủ Đề"] for info in st.session_state.selected_bai_info],
                "Yêu Cầu Cần Đạt": [info["Yêu Cầu Cần Đạt"] for info in st.session_state.selected_bai_info],
                "Số lượng câu hỏi": {
                    mucdo: sum(info["Số lượng câu hỏi"].get(mucdo, 0) for info in st.session_state.selected_bai_info)
                    for mucdo in st.session_state.questions_per_mucdo.keys()
                }
            }
            print('---------------------------------------------------------------- end')
            print(exam_params)
            try:
                app.questions = get_questions(
                    topics="",
                    number_of_questions=total_questions,
                    number_of_answers=number_of_answers,
                    sach=exam_params["Sách"],  # Flatten list
                    bai=exam_params["Bài"],
                    chude=exam_params["Chủ Đề"],  # Flatten list
                    mucdo=[(mucdo, count) for mucdo, count in exam_params["Số lượng câu hỏi"].items() if count > 0],
                    yccd=[yccd for sublist in exam_params["Yêu Cầu Cần Đạt"] for yccd in sublist] , # Flatten list
                    contains_num=3,
                    delta=0.5
                )
            except Exception as e:
                st.error(f"An error occurred while generating the questions: {str(e)}")

        if st.button("Tạo đề thi từ việc kết hợp AI và ngân hàng câu hỏi", help="Tạo câu hỏi theo các thông số"):
            st.warning("Hệ thống đang tạo đề thi. Bạn vui lòng chờ đợi....")

            exam_params = {
                "Sách": [info["Sách"] for info in st.session_state.selected_bai_info],
                "Bài": [info["Bài"] for info in st.session_state.selected_bai_info],
                "Chủ Đề": [info["Chủ Đề"] for info in st.session_state.selected_bai_info],
                "Yêu Cầu Cần Đạt": [info["Yêu Cầu Cần Đạt"] for info in st.session_state.selected_bai_info],
                "Số lượng câu hỏi": {
                    mucdo: sum(info["Số lượng câu hỏi"].get(mucdo, 0) for info in st.session_state.selected_bai_info)
                    for mucdo in st.session_state.questions_per_mucdo.keys()
                }
            }
            try:
                app.questions = get_questions_from_bank(
                    topics="",
                    number_of_questions=total_questions,
                    number_of_answers=number_of_answers,
                    sach=exam_params["Sách"],  # Flatten list
                    bai=exam_params["Bài"],
                    chude=exam_params["Chủ Đề"],  # Flatten list
                    mucdo=[(mucdo, count) for mucdo, count in exam_params["Số lượng câu hỏi"].items() if count > 0],
                    yccd=[yccd for sublist in exam_params["Yêu Cầu Cần Đạt"] for yccd in sublist], # Flatten list
                    contains_num=3,
                    delta=0.5  # Flatten list
                )

            except Exception as e:
                st.error(f"An error occurred while generating the questions: {str(e)}")


        if app.questions is not None:
            st.info(
                f" Một bài kiểm tra có {len(app.questions)} câu hỏi đã được tạo.  Bạn có thể tải xuống câu hỏi dưới dạng tệp PDF hoặc làm bài kiểm tra trong ứng dụng."
            )
            left, center, right = st.columns(3)

            with left:
                questions_to_pdf(app.questions, "questions.pdf")
                with open("questions.pdf", "rb") as pdf_file:
                    st.download_button(
                        label="Tải đề thi xuống",
                        data=pdf_file.read(),
                        file_name="questions.pdf",
                        mime="application/pdf",
                        help="Tải đề thi dưới dạng pdf"
                    )

            with center:
                if st.button("Lưu đề thi", help="Khi chọn chức năng này, hệ thống sẽ lưu lại đề thi của bạn. Nhằm mục đích đa dạng hoá các đề thi sinh ra tiếp theo tránh trùng lặp"):
                    # Prepare the exam data to save
                    if app.questions is not None:  # Ensure app.questions is not None
                        # Create a new entry with the current timestamp and questions
                        new_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Add timestamp
                            "questions": [question.to_dict() for question in app.questions]
                            # Assuming to_dict method exists
                        }

                        # Initialize existing data
                        existing_data = []

                        # Check if the file exists
                        if os.path.exists("exam.json"):
                            try:
                                # Load existing data
                                with open("exam.json", "r", encoding="utf-8") as f:
                                    # Check if the file is empty
                                    if os.stat("exam.json").st_size == 0:
                                        existing_data = []  # File is empty
                                    else:
                                        existing_data = json.load(f)  # Load existing data
                            except json.JSONDecodeError:
                                print("Error: The JSON file is corrupted or empty. Initializing new data.")
                                existing_data = []  # Reset to an empty list if there's an error

                        # Append the new entry
                        existing_data.append(new_entry)

                        # Save back to JSON file
                        with open("exam.json", "w", encoding="utf-8") as f:
                            json.dump(existing_data, f, ensure_ascii=False, indent=4)

                        st.success("Đề thi đã được lưu thành công vào exam.json.")
                    else:
                        st.error("Không có câu hỏi nào để lưu.")

            with right:
                if st.button("Làm bài thi trực tiếp", help="Làm bài thi trực tiếp"):
                    app.change_page(PageEnum.QUESTIONS)


class QuestionsPage(Page):

    def __init__(self):
        self.number_of_question = 0

    def render(self, app):
        """
        Render the page
        """
        st.title("BÀI THI TRỰC TIẾP")
        
        # Kiểm tra nếu không có câu hỏi
        if not app.questions or len(app.questions) == 0:
            st.error("Không có câu hỏi nào được tạo. Vui lòng quay lại để tạo câu hỏi.")
            return

        # Kiểm tra chỉ mục hợp lệ
        if self.number_of_question < 0 or self.number_of_question >= len(app.questions):
            st.error("Không thể truy cập câu hỏi. Chỉ mục không hợp lệ.")
            return

        question = app.questions[self.number_of_question]

        answer = self.__render_question(question, app.get_answer(self.number_of_question))

        app.add_answer(self.number_of_question, answer)

        left, center, right = st.columns(3)

        if self.number_of_question != 0:
            with left:
                if st.button("Trở về", help="Trở về câu hỏi trước"):
                    self.__change_question(self.number_of_question - 1)

        with center:
            if st.button("Nộp bài", help="Nộp bài thi và đi đến trang kết quả"):
                app.change_page(PageEnum.RESULTS)

        if self.number_of_question != len(app.questions) - 1:
            with right:
                if st.button("Câu tiếp theo", help="Di chuyển đến câu hỏi tiếp theo"):
                    self.__change_question(self.number_of_question + 1)

    @staticmethod
    def __render_question(question: Question, index_answer: Optional[int]) -> int:
        """
        Render a question and return the index of the answer selected by the user
        :param question: Question to render
        :param index_answer: Index of the answer selected by the user (if any)
        :return: Index of answer selected by the user
        """
        if index_answer is None:
            index_answer = 0

        st.write(f"**{question.id}. {question.question}**")
        answer = st.radio("Answer", question.answers, index=index_answer)

        index = question.answers.index(answer)

        return index

    def __change_question(self, index: int):
        """
        Change the current question and rerun the app
        :param index: Index of the question to change to
        """
        self.number_of_question = index
        st.rerun()


class ResultsPage:

    def __init__(self):
        self.clarifications = {}

    def render(self, app):
        """
        Render the page
        """
        st.title("Kết quả")

        num_correct = self.__get_correct_answers(app)

        st.write(f"### Số lượng câu hỏi: {len(app.questions)}")
        st.write(f"### Số câu trả lời đúng: {num_correct}")
        st.write(f"### Tỉ lệ trả lời đúng: {num_correct / len(app.questions) * 100:.2f}%")

        for index, question in enumerate(app.questions):
            self.__render_question(question, app.get_answer(index))

        left, right = st.columns(2)

        with left:

            if st.button("Tạo đề thi mới"):
                app.reset()
                app.change_page(PageEnum.GENERATE_EXAM)

        with right:

            questions_to_pdf(app.questions, "questions.pdf")
            st.download_button(
                "Download",
                data=open("questions.pdf", "rb").read(),
                file_name="questions.pdf",
                mime="application/pdf",
                help="Tải đề thi dưới dạng PDF"
            )

    def __render_question(self, question: Question, user_answer: int):
        """
        Render a question with the correct answer
        :param question: Question to render
        :param user_answer: Index of the answer selected by the user
        """
        st.write(f"**{question.id}. {question.question}**")

        if question.correct_answer == user_answer:
            for index, answer in enumerate(question.answers):
                if index == user_answer:
                    st.write(f":green[{chr(ord('a') + index)}) {answer}]")
                else:
                    st.write(f"{chr(ord('a') + index)}) {answer}")

        else:
            for index, answer in enumerate(question.answers):
                if index == user_answer:
                    st.write(f":red[{chr(ord('a') + index)}) {answer}]")
                elif index == question.correct_answer:
                    st.write(f":green[{chr(ord('a') + index)}) {answer}]")
                else:
                    st.write(f"{chr(ord('a') + index)}) {answer}")

        clarify_button = st.button(
            "Giải thích",
            help="Get more information about the question",
            key=f"clarify_question_{question.id}"
        )

        if not clarify_button:
            return

        if question.id not in self.clarifications:
            st.warning("This can take a while...")
            self.clarifications[question.id] = clarify_question(question)

        st.write(self.clarifications[question.id])

    @staticmethod
    def __get_correct_answers(app):
        """
        Get the number of correct answers
        :param app: App instance
        :return: Number of correct answers
        """
        correct_answers = 0
        for index, question in enumerate(app.questions):
            if question.correct_answer == app.get_answer(index):
                correct_answers += 1

        return correct_answers
