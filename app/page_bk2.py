import os
import pandas as pd
import streamlit as st
from abc import abstractmethod
from typing import Optional
from typing import List
from model.question import Question
from utils.api import get_questions, clarify_question, get_questions_from_bank
from utils.generate_document import questions_to_pdf

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

class GenerateExamPage(Page):

    description = """
    <p>Chào mừng bạn đến với hệ thống tạo đề trắc nghiệm thông minh, một công cụ hỗ trợ giáo viên trong việc thiết kế và quản lý các đề thi một cách hiệu quả!</p>

    <p>Tính năng nổi bật của hệ thống:</p>
    <ul>
        <li>Hỗ trợ giáo viên dễ dàng tạo đề thi: Với giao diện thân thiện và dễ sử dụng, giáo viên có thể nhanh chóng tạo ra các đề thi trắc nghiệm phù hợp với nhiều cấp độ học sinh và chủ đề khác nhau.</li>
        <li>Tối ưu hóa quá trình đánh giá học sinh: Thông qua các phân tích dữ liệu, giáo viên có thể dễ dàng theo dõi kết quả học tập của học sinh và điều chỉnh phương pháp giảng dạy cho phù hợp.</li>
    </ul>
    """

    def render(self, app):
        """
        Render the page
        """
        # Thay đổi màu nền của trang và thêm border
        st.markdown(
            """
            <style>
            .main {
                background-color: #f7f9fc; /* Màu nền nhẹ hơn */
                padding: 20px; /* Tăng padding */
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Thêm bóng */
                border: 2px solid #4A90E2; /* Thêm border cho nội dung chính */
            }
            h1 {
                text-align: center;
                font-size: 2.5em;
                font-weight: bold;
                color: #4A90E2;
                margin-bottom: 10px; /* Khoảng cách dưới tiêu đề */
            }
            h2 {
                text-align: center;
                color: #555555;
                margin-bottom: 20px; /* Khoảng cách dưới tiêu đề phụ */
            }
            p {
                font-size: 1.1em;
                line-height: 1.6;
                margin: 15px 0; /* Khoảng cách giữa các đoạn văn */
            }
            </style>
            """, unsafe_allow_html=True
        )

        # Tiêu đề trang trí
        st.markdown("""
            <h1>Hệ thống thông minh tạo đề kiểm tra trắc nghiệm môn Tin học 10</h1>
        """, unsafe_allow_html=True)

        # Mô tả với định dạng HTML
        st.markdown(self.description, unsafe_allow_html=True)

        # Load CSV data
        csv_file_path = '/mount/src/exam-generator/data.csv'  # Update this with your CSV path
        data = pd.read_csv(csv_file_path, delimiter=';')
        data = data[['SACH', 'BAI', 'CHUDE', 'MUCDO', 'NOIDUNG_YCCD']].dropna()  # Chỉ định cột NOIDUNG_YCCD

        # Create combo box to select 'Sách'
        st.header("Tạo đề AI")
        sach = st.selectbox("Sách", data['SACH'].unique(), key="sach_select")

        # Filter 'Bài' based on selected 'Sách'
        bai_options = data[data['SACH'] == sach]['BAI'].unique()
        selected_bai = st.multiselect("Bài", bai_options, key="bai_multiselect")

        # Filter 'YCCD' based on selected 'Bài'
        if selected_bai:
            yccd_options = data[data['BAI'].isin(selected_bai)]['NOIDUNG_YCCD'].unique()
            selected_yccd = st.multiselect("Yêu cầu cần đạt", yccd_options, key="yccd_multiselect")
        else:
            selected_yccd = []

        # Filter 'Chủ Đề' based on selected 'Bài'
        if selected_bai:
            chude_options = data[(data['SACH'] == sach) & (data['BAI'].isin(selected_bai))]['CHUDE'].unique()
            selected_chude = st.multiselect("Chủ Đề", chude_options, key="chude_multiselect")
        else:
            selected_chude = []

        # Select multiple 'Mức Độ' and input the number of questions for each level
        st.header("Nhập số lượng câu hỏi theo từng mức độ")
        mucdo_options = data['MUCDO'].unique()
        questions_per_mucdo = {}

        # Input for each Mức Độ
        for mucdo in mucdo_options:
            questions_per_mucdo[mucdo] = st.number_input(
                f"Số lượng câu hỏi cho mức độ '{mucdo}'",
                min_value=0,
                value=0,
                step=1,
                key=f"questions_{mucdo}"
            )

        # Calculate the total number of questions based on user input
        total_questions = sum(questions_per_mucdo.values())

        # Display total number of questions (calculated based on user input)
        st.markdown(f"### Tổng số câu hỏi: {total_questions}")

        # Set the number of answers based on total number of questions
        number_of_answers = st.number_input(
            "Number of answers",
            min_value=3,
            max_value=5,
            value=4,
            help=f"Tổng số câu hỏi: {total_questions}"
        )

        # Button to submit the form and send selected information
        if st.button("Sinh câu hỏi từ ChatGPT", help="Generate the questions according to the parameters"):
            st.warning("Generating questions. This may take a while...")

            # Display the selected values
            st.write(f"Selected Sách: {sach}")
            st.write(f"Selected Bài: {', '.join(selected_bai)}")
            st.write(f"Selected Chủ Đề: {', '.join(selected_chude)}")

            # Display the selected Mức Độ and number of questions for each level
            for mucdo, count in questions_per_mucdo.items():
                st.write(f"Số lượng câu hỏi cho mức độ '{mucdo}': {count}")

            try:
                # Flatten the questions_per_mucdo dictionary to a list format [(mucdo, count), ...]
                mucdo_with_counts = [(mucdo, count) for mucdo, count in questions_per_mucdo.items() if count > 0]

                # Pass selected values to the get_questions function
                app.questions = get_questions(
                    topics="",
                    number_of_questions=total_questions,  # Pass the total number of questions
                    number_of_answers=number_of_answers,  # Total number of answers per question
                    sach=sach,
                    bai=selected_bai,
                    chude=selected_chude,
                    mucdo=mucdo_with_counts  # Pass the list of Mức Độ with their respective counts
                )
            except Exception as e:
                st.error(f"An error occurred while generating the questions: {str(e)}")

        # Button to generate questions from the bank
        if st.button("Sinh câu hỏi từ Ngân hàng câu hỏi", help="Generate the questions according to the parameters"):
            st.warning("Generating questions. This may take a while...")

            # Display the selected values
            st.write(f"Selected Sách: {sach}")
            st.write(f"Selected Bài: {', '.join(selected_bai)}")
            st.write(f"Selected Chủ Đề: {', '.join(selected_chude)}")

            # Display the selected Mức Độ and number of questions for each level
            for mucdo, count in questions_per_mucdo.items():
                st.write(f"Số lượng câu hỏi cho mức độ '{mucdo}': {count}")

            try:
                # Flatten the questions_per_mucdo dictionary to a list format [(mucdo, count), ...]
                mucdo_with_counts = [(mucdo, count) for mucdo, count in questions_per_mucdo.items() if count > 0]

                # Pass selected values to the get_questions_from_bank function
                app.questions = get_questions_from_bank(
                    topics="",
                    number_of_questions=total_questions,  # Pass the total number of questions
                    number_of_answers=number_of_answers,  # Total number of answers per question
                    sach=sach,
                    bai=selected_bai,
                    chude=selected_chude,
                    mucdo=mucdo_with_counts  # Pass the list of Mức Độ with their respective counts
                )
            except Exception as e:
                st.error(f"An error occurred while generating the questions: {str(e)}")

        # Display results or proceed with the next steps
        if app.questions is not None:
            st.info(
                f"An exam with {len(app.questions)} questions has been generated. You "
                f"can download the questions as a PDF file or take the exam in the app."
            )

            left, center, right = st.columns(3)

            with left:
                questions_to_pdf(app.questions, "questions.pdf")
                with open("questions.pdf", "rb") as pdf_file:
                    st.download_button(
                        label="Download",
                        data=pdf_file.read(),
                        file_name="questions.pdf",
                        mime="application/pdf",
                        help="Download the questions as a PDF file"
                    )

            with right:
                if st.button("Start exam", help="Start the exam"):
                    app.change_page(PageEnum.QUESTIONS)


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
