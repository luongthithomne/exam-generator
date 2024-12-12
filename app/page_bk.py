import os
import pandas as pd
import streamlit as st
from abc import abstractmethod
from typing import Optional
from typing import List
from model.question import Question
from utils.api import get_questions, clarify_question, get_questions_from_bank
from utils.generate_document import questions_to_pdf

csv_file_path = '/mount/src/exam-generator/data.csv'  # Update this with your CSV path


class PageEnum:
    """
    Enum for pages
    """
    GENERATE_EXAM = 0
    QUESTIONS = 1
    RESULTS = 2


class Page:

    @abstractmethod
    def render(self, app):
        """
        Render the page (must be implemented by subclasses)
        """


class GenerateExamPage:

    def render(self, app):
        # Load CSV data
        data = pd.read_csv(csv_file_path, delimiter=';')
        data = data[['SACH', 'BAI', 'CHUDE', 'MUCDO', 'NOIDUNG_YCCD']].dropna()

        # Khởi tạo dữ liệu lưu trữ các dòng đã tạo
        if 'rows' not in st.session_state:
            st.session_state['rows'] = []

        # Điều chỉnh style cho giao diện full width
        st.markdown("""
            <style>
            body {
                width: 100%;
            }
            .st-emotion-cache-13ln4jf {
                max-width: 100% !important;
            }
            .main {
                max-width: 100%;
                padding-left: 5%;
                padding-right: 5%;
            }
            </style>
        """, unsafe_allow_html=True)

        # Tạo hai cột chiếm đều nhau màn hình
        col1, col2 = st.columns(2)

        # Cột trái: Form để thêm hàng mới
        with col1:
            with st.form(key="form_add_row"):
                st.markdown("### Thêm hàng mới")
                row_id = len(st.session_state['rows'])

                # Chọn sách
                sach = st.selectbox(f"Sách {row_id + 1}", data['SACH'].unique(), key=f"sach_{row_id}_new")

                # Lọc các tùy chọn Bài dựa trên sách đã chọn
                if sach:
                    bai_options = data[data['SACH'] == sach]['BAI'].unique()
                else:
                    bai_options = []
                bai = st.selectbox(f"Bài {row_id + 1}", bai_options, key=f"bai_{row_id}_new")

                # Lọc các tùy chọn Chủ đề dựa trên sách và bài đã chọn
                if bai:
                    chude_options = data[(data['SACH'] == sach) & (data['BAI'] == bai)]['CHUDE'].unique()
                else:
                    chude_options = []
                chude = st.selectbox(f"Chủ đề {row_id + 1}", chude_options, key=f"chude_{row_id}_new")

                # Chọn mức độ từ dữ liệu gốc
                mucdo_options = data["MUCDO"].unique() if bai else []
                mucdo = st.selectbox(f"Mức độ {row_id + 1}", mucdo_options, key=f"mucdo_{row_id}_new")

                # Hiển thị YCCD với phiên bản rút gọn nhưng lưu giữ giá trị gốc
                if bai:
                    yccd_options = data[(data["BAI"] == bai)]["NOIDUNG_YCCD"].unique()
                else:
                    yccd_options = []

                truncated_yccd_options = [y[:20] + "..." if len(y) > 20 else y for y in yccd_options]
                yccd = st.multiselect(f"Yêu cầu cần đạt {row_id + 1}", truncated_yccd_options, key=f"yccd_{row_id}_new")

                # Submit form để thêm dòng mới
                if st.form_submit_button("Thêm hàng mới"):
                    original_yccd = [yccd_options[truncated_yccd_options.index(y)] for y in yccd]

                    # Lưu thông tin hàng mới vào session_state
                    new_row = {
                        "SACH": sach,
                        "BAI": bai,
                        "CHUDE": chude,
                        "MUCDO": mucdo,
                        "YCCD": original_yccd  # Lưu dữ liệu gốc của YCCD
                    }
                    st.session_state['rows'].append(new_row)

        # Cột phải: Hiển thị bảng các hàng đã tạo
        with col2:
            st.markdown("""
                <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 10px;
                    text-align: left;
                    border: 1px solid #dddddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown("""
                <table>
                    <thead>
                        <tr>
                            <th>Sách</th>
                            <th>Bài</th>
                            <th>Chủ đề</th>
                            <th>Mức độ</th>
                            <th>Yêu cầu cần đạt</th>
                        </tr>
                    </thead>
                    <tbody id="table_body">
            """, unsafe_allow_html=True)

            # Hiển thị tất cả các dòng đã tạo
            for row in st.session_state['rows']:
                sach = row["SACH"]
                bai = row["BAI"]
                chude = row["CHUDE"]
                mucdo = row["MUCDO"]
                yccd = row["YCCD"]

                # Cắt chuỗi YCCD và hiển thị chỉ với 20 ký tự đầu tiên
                truncated_yccd = [y[:20] + "..." if len(y) > 20 else y for y in yccd]

                st.markdown(f"""
                    <tr>
                        <td>{sach}</td>
                        <td>{bai}</td>
                        <td>{chude}</td>
                        <td>{mucdo}</td>
                        <td>{', '.join(truncated_yccd)}</td>
                    </tr>
                """, unsafe_allow_html=True)

            st.markdown("</tbody></table>", unsafe_allow_html=True)

        # Button để tổng hợp tất cả thông tin từ bảng
        if st.button("Tổng hợp thông tin"):
            result_dict = st.session_state['rows']
            st.json(result_dict)

        # Tạo phần submit để lấy thông tin tổng hợp từ bảng
        if st.button("Sinh câu hỏi"):
            st.success("Câu hỏi đã được sinh thành công!")

class QuestionsPage(Page):

    def __init__(self):
        self.number_of_question = 0

    def render(self, app):
        """
        Render the page
        """
        st.title("Questions")

        question = app.questions[self.number_of_question]

        answer = self.__render_question(question, app.get_answer(self.number_of_question))

        app.add_answer(self.number_of_question, answer)

        left, center, right = st.columns(3)

        if self.number_of_question != 0:
            with left:
                if st.button("Previous", help="Go to the previous question"):
                    self.__change_question(self.number_of_question - 1)

        with center:
            if st.button("Finish", help="Finish the exam and go to the results page"):
                app.change_page(PageEnum.RESULTS)

        if self.number_of_question != len(app.questions) - 1:
            with right:
                if st.button("Next", help="Go to the next question"):
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
        st.title("Results")

        num_correct = self.__get_correct_answers(app)

        st.write(f"### Number of questions: {len(app.questions)}")
        st.write(f"### Number of correct answers: {num_correct}")
        st.write(f"### Percentage of correct answers: {num_correct / len(app.questions) * 100:.2f}%")

        for index, question in enumerate(app.questions):
            self.__render_question(question, app.get_answer(index))

        left, right = st.columns(2)

        with left:

            if st.button("Generate new exam"):
                app.reset()
                app.change_page(PageEnum.GENERATE_EXAM)

        with right:

            questions_to_pdf(app.questions, "questions.pdf")
            st.download_button(
                "Download",
                data=open("questions.pdf", "rb").read(),
                file_name="questions.pdf",
                mime="application/pdf",
                help="Download the questions as a PDF file"
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
