import json
from datetime import datetime, timedelta
import os
from typing import Tuple
import openai
import re
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model.question import Question
import pandas as pd
from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import random

ROOT_PATH = '/mount/src/exam-generator'
MODEL = "gpt-4o-mini"
# Load PhoBERT model and tokenizer
tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


# # Load pre-trained Word2Vec model
# from gensim.downloader import load
#
# word2vec_model = load("word2vec-google-news-300")
# # Load BERT model
# tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')


def complete_text(prompt: str) -> str:
    """
    Complete text using GPT-3.5 Turbo
    :param prompt: Prompt to complete
    :return: Completed text
    """

    messages = [{"role": "user", "content": prompt}]

    return openai.ChatCompletion.create(model=MODEL, messages=messages)["choices"][0]["message"]["content"]


def prepare_prompt(topics: str, number_of_questions: int, number_of_answers: int,
                   sach: List[str], bai: List[str], chude: List[str],
                   mucdo: List[Tuple[str, int]], yccd: List[str]) -> str:
    bai_str = ', '.join(bai)
    chude_str = ', '.join(chude)
    yccd_str = ', '.join(yccd)
    # Convert mucdo list of tuples to a string format for the prompt
    mucdo_str = ', '.join([f"{count} câu hỏi mức độ {level}" for level, count in mucdo])

    prompt = (
        f"Tạo một đề thi trắc nghiệm gồm {number_of_questions} câu hỏi với {number_of_answers} câu trả lời "
        f"Đảm bảo các câu hỏi rõ ràng và không chứa định dạng sai như thừa ký tự hay dấu không cần thiết. "
        f"Ví dụ câu hỏi: 'Những câu phát biểu nào sau đây đúng?' phải được viết rõ ràng và có đáp án hợp lệ."
        f"cho mỗi câu hỏi. Đáp án đúng phải được bôi đậm (bao quanh bởi **). "
        f"Chủ đề của đề thi là Tin học "
        f"Sách được chọn: {', '.join(sach)}, các bài học: {bai_str}, các chủ đề: {chude_str}, "
        f"mức độ: {mucdo_str}, và yêu cầu cần đạt: {yccd_str}. "
        f"Chỉ tạo câu hỏi và câu trả lời, không tạo đề thi hoàn chỉnh và không hiển thị câu nào là mức độ nào.Sau khi tạo lần 1 thì dựa vào những câu hỏi vừa tạo đó tạo ra những câu hỏi tương tự những câu hỏi đã tạo lần 1. Không cần hiển thị câu hỏi lần 1. Chỉ cần hiển thị câu hỏi lần 2 và không cần hiển thị tiêu đề là câu hỏi tạo lần 2. Chỉ cần hiển thị câu hỏi và câu trả lời đúng format là được."
    )
    print(prompt)
    print('------------------')
    return prompt


def prepare_prompt_regenerate(questions: List[Question]) -> str:
    prompt = "Dựa trên các câu hỏi đã có, hãy tạo ra các câu hỏi tương đương với cùng thông tin nhưng được diễn đạt khác. Dưới đây là câu hỏi và câu trả lời:\n\n"
    for q in questions:
        prompt += f"{q.question}\n"
        for i, answer in enumerate(q.answers):
            if i == q.correct_answer:
                prompt += f"**{answer}**\n"
            else:
                prompt += f"{answer}\n"
        prompt += "\n"
    prompt += "Đảm bảo các câu hỏi rõ ràng và không chứa định dạng sai như thừa ký tự hay dấu không cần thiết. Ví dụ câu hỏi: 'Những câu phát biểu nào sau đây đúng?' phải được viết rõ ràng và có đáp án hợp lệ. cho mỗi câu hỏi. Đáp án đúng phải được bôi đậm (bao quanh bởi **). Chỉ tạo câu hỏi và câu trả lời, không tạo đề thi hoàn chỉnh. Chỉ cần hiển thị câu hỏi và câu trả lời đúng format là được."
    return prompt
    # return (
    #     f"Tạo một đề thi trắc nghiệm gồm {number_of_questions} câu hỏi với {number_of_answers} câu trả lời "
    #     f"cho mỗi câu hỏi. Đáp án đúng phải được bôi đậm (bao quanh bởi **). "
    #     f"Chủ đề của đề thi là Tin học "
    #     f"Sách được chọn: {', '.join(sach)}, các bài học: {bai_str}, các chủ đề: {chude_str}, "
    #     f"mức độ: {mucdo_str}, và yêu cầu cần đạt: {yccd_str}. "
    #     f"Chỉ tạo câu hỏi và câu trả lời, không tạo đề thi hoàn chỉnh và không hiển thị câu nào là mức độ nào.Sau khi tạo lần 1 thì dựa vào những câu hỏi vừa tạo đó tạo ra những câu hỏi tương tự những câu hỏi đã tạo lần 1. Không cần hiển thị câu hỏi lần 1. Chỉ cần hiển thị câu hỏi lần 2 và không cần hiển thị tiêu đề là câu hỏi tạo lần 2. Chỉ cần hiển thị câu hỏi và câu trả lời đúng format là được."
    # )


def sanitize_line(line: str, is_question: bool) -> str:
    """
    Sanitize a line from the response
    :param line: Line to sanitize
    :param is_question: Whether the line is a question or an answer
    :return: Sanitized line
    """
   # if is_question:
        #line = re.sub(r"^Q:\s?", "", line)  # Xóa tiền tố "Q: " nếu tồn tại
        #new_line = re.sub(r"[0-9]+.", " ", line, count=1) # Xóa số thứ tự câu hỏi
    #else:
        #new_line = re.sub(r"[a-eA-E][).]", " ", line, count=1) #xóa tiền tố đáp án 

    #return new_line
    if is_question:
        line = re.sub(r"(Câu\s+\d+:)\s*(Câu\s+\d+:)?", "Câu ", line)  # Xóa chuỗi lặp "Câu X: Câu Y"
        line = re.sub(r"^Q:\s?", "", line)  # Xóa tiền tố "Q: "
        line = re.sub(r"[0-9]+\. ", "", line)  # Xóa số thứ tự nếu có
    else:
        # Loại bỏ tiền tố lặp lại (ví dụ: "C. C.")
        line = re.sub(r"^[a-dA-D][.)]\s*", "", line)  # Xóa tiền tố đáp án (A., B., ...)
        line = re.sub(r"^[a-dA-D][.)]\s*", "", line)  # Nếu còn lặp thêm lần nữa, xóa tiếp
    return line.strip()


def get_correct_answer(answers: List[str]) -> int:
    """
    Return the index of the correct answer
    :param answers: List of answers
    :return: Index of the correct answer if found, -1 otherwise
    """
    for index, answer in enumerate(answers):
        if answer.count("**") > 0:
            return index

    return -1


# def response_to_questions(response: str) -> List[Question]:
#     """Convert the response from the API to a list of questions."""
#     questions = []
#     count = 1
#
#     for question_text in response.split("\n\n"):
#         question_text = question_text.strip()
#         if not question_text:
#             continue
#
#         question_lines = question_text.splitlines()
#         question = sanitize_line(question_lines[0], is_question=True)
#         answers = list(map(lambda line: sanitize_line(line, is_question=False), question_lines[1:]))
#         correct_answer = get_correct_answer(answers)
#         answers[correct_answer] = answers[correct_answer].replace("**", "")
#         answers = list(map(lambda answer: answer.strip(), answers))
#
#         questions.append(Question(count, question, answers, correct_answer))
#         count += 1
#
#     return questions


"""
app.questions = get_questions(
                    topics="",
                    number_of_questions=total_questions,
                    number_of_answers=number_of_answers,
                    sach=exam_params['Sách'],
                    bai=exam_params['Bài'],
                    chude=exam_params['Chủ Đề'],
                    mucdo=[(mucdo, count) for mucdo, count in exam_params['Số lượng câu hỏi'].items() if count > 0],
                    yccd= exam_params['Yêu Cầu Cần Đạt']
                )
"""


# def get_questions(topics: str, number_of_questions: int, number_of_answers: int,
#                   sach: List[str], bai: List[str], chude: List[str], mucdo: List[Tuple[str, int]],
#                   yccd: List[str], contains_num: int, delta: float) -> List[Question]:
#     print('get_questions')
#
#     recent_questions = load_recent_questions(
#         "ROOT_PATH + /exam.json")
#     prompt = prepare_prompt(topics, number_of_questions, number_of_answers, sach, bai, chude, mucdo, yccd)
#     response = complete_text(prompt)
#     new_questions = response_to_questions(response)
#
#     flagcontinue, contains_num = get_lowest_similarity_exam(new_questions, recent_questions, delta, contains_num)
#     if flagcontinue == [] and contains_num != 0:
#         get_questions(topics, number_of_questions, number_of_answers, sach, bai, chude, mucdo, yccd, contains_num, 0.5)
#     else:
#         return new_questions
#     return new_questions

def get_questions(topics: str, number_of_questions: int, number_of_answers: int,
                  sach: List[str], bai: List[str], chude: List[str], mucdo: List[Tuple[str, int]],
                  yccd: List[str], contains_num: int, delta: float) -> List[Question]:
    print('get_questions')

    data = pd.read_csv(ROOT_PATH + '/data.csv', delimiter=';', on_bad_lines='skip')    
    # Filter the data based on selected Sách, Bài, and Chủ Đề
    filtered_data = data[
        (data['SACH'].isin(sach)) &  # Changed from == to isin for list of Sách
        (data['BAI'].isin(bai)) &
        (data['CHUDE'].isin(chude))
        ]


    questions = []
    for mucdo_level, count in mucdo:
        # Filter for the current Mức Độ
        mucdo_filtered_data = filtered_data[filtered_data['MUCDO'] == mucdo_level]
        # Extract the list of questions (CAUHOI) and answers (CAUTRALOI) for this Mức Độ
        available_questions = mucdo_filtered_data[['CAUHOI', 'CAUTRALOI', 'DAPAN']].values.tolist()

        # If there are enough available questions, randomly select the required amount
        selected_questions = random.sample(available_questions, min(len(available_questions), count))

        # Create Question objects from the selected data
        for i, (cauhoi, cautraloi, dapan) in enumerate(selected_questions):
            cauhoi = sanitize_line(cauhoi,is_question=True)  # Làm sạch câu hỏi
            answers = cautraloi.split('~')  # Tách đáp án bởi dấu ~
            # Find the most similar answer to the correct answer using PhoBERT embeddings
            correct_answer_index = get_most_similar_answer(dapan, answers)

            # Create the Question object
            question = Question(
                id=len(questions) + 1,
                question=cauhoi,
                answers=answers,
                correct_answer=correct_answer_index
            )
            questions.append(question)
    
    if len(questions) < number_of_questions:
        additional_needed = number_of_questions - len(questions)
        additional_questions = random.sample(filtered_data[['CAUHOI', 'CAUTRALOI', 'DAPAN']].values.tolist(),
                                             additional_needed)
        for cauhoi, cautraloi, dapan in additional_questions:
         
            answers = cautraloi.split('~')
            correct_answer_index = get_most_similar_answer(dapan, answers)
            question = Question(
                id=len(questions) + 1,
                question=cauhoi,
                answers=answers,
                correct_answer=correct_answer_index
            )
            questions.append(question)

    recent_questions = load_recent_questions(
        ROOT_PATH + "/exam.json")

    flagcontinue, contains_num = get_lowest_similarity_exam(questions, recent_questions, delta, contains_num)
    if flagcontinue == [] and contains_num != 0:
        get_questions(topics, number_of_questions, number_of_answers, sach, bai, chude, mucdo, yccd,
                                contains_num, 0.5)
    else:
        new_ques = []
        for ques in questions:
            quesv2 = Question(id=ques.id
                              ,question=sanitize_line(ques.question,is_question=True)
                              ,answers=ques.answers
                              ,correct_answer=ques.correct_answer)
            new_ques.append(quesv2)
        return new_ques
    return []


def clarify_question(question: Question) -> str:
    """
    Clarify a question using GPT-3.5 Turbo
    :param question: Question to clarify
    :return: Text clarifying the question
    """
    join_questions = "\n".join([f"{chr(ord('a') + i)}. {answer}" for i, answer in enumerate(question.answers)])

    prompt = f"Với câu hỏi: {question.question}\n"
    prompt += f" và câu trả lời: {join_questions}\n\n"
    prompt += f"Tại sao câu trả lời đúng là {chr(ord('a') + question.correct_answer)}?\n\n"

    return complete_text(prompt)


def get_embedding(text: str):
    """
    Get the embedding for the input text using PhoBERT.
    :param text: Input text
    :return: Embedding tensor
    """
    inputs = tokenizer_phobert(text, return_tensors="pt", max_length=256, truncation=True)
    with torch.no_grad():
        outputs = phobert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Get the mean of the last hidden state as the embedding


def get_most_similar_answer(correct_answer: str, answers: List[str]) -> int:
    """
    Use PhoBERT embeddings to find the most similar answer to the correct answer.
    :param correct_answer: The correct answer as provided in the data
    :param answers: List of possible answers
    :return: The index of the answer most similar to the correct answer
    """
    # Get the embedding for the correct answer
    correct_answer_emb = get_embedding(correct_answer)

    # Calculate similarities with each answer
    similarities = []
    for answer in answers:
        answer_emb = get_embedding(answer)
        similarity = F.cosine_similarity(correct_answer_emb, answer_emb)
        similarities.append(similarity.item())

    # Get the index of the answer with the highest similarity
    most_similar_index = similarities.index(max(similarities))

    return most_similar_index


def get_questions_from_bank(topics: str, number_of_questions: int, number_of_answers: int,
                            sach: List[str], bai: List[str], chude: List[str], mucdo: List[tuple],
                            yccd: List[str], contains_num: int, delta: float) -> List[Question]:
    # Load the data from CSV
    data = pd.read_csv(ROOT_PATH + '/data.csv', delimiter=';', on_bad_lines='skip')
    new_questions = []
    # Filter the data based on selected Sách, Bài, and Chủ Đề
    filtered_data = data[
        (data['SACH'].isin(sach)) &  # Changed from == to isin for list of Sách
        (data['BAI'].isin(bai)) &
        (data['CHUDE'].isin(chude))
        ]

    questions = []

    for mucdo_level, count in mucdo:
        # Filter for the current Mức Độ
        mucdo_filtered_data = filtered_data[filtered_data['MUCDO'] == mucdo_level]

        # Extract the list of questions (CAUHOI) and answers (CAUTRALOI) for this Mức Độ
        available_questions = mucdo_filtered_data[['CAUHOI', 'CAUTRALOI', 'DAPAN']].values.tolist()

        # If there are enough available questions, randomly select the required amount
        selected_questions = random.sample(available_questions, min(len(available_questions), count))

        # Create Question objects from the selected data
        for i, (cauhoi, cautraloi, dapan) in enumerate(selected_questions):
            # Làm sạch câu hỏi
            cauhoi = sanitize_line(cauhoi, is_question=True)
            # Làm sạch từng đáp án
            answers = [sanitize_line(answer.strip(), is_question=False) for answer in cautraloi.split('~')]

            # Find the most similar answer to the correct answer using PhoBERT embeddings
            correct_answer_index = get_most_similar_answer(dapan, answers)

            # Create the Question object
            question = Question(
                id=len(questions) + 1,
                question=cauhoi,
                answers=answers,
                correct_answer=correct_answer_index
            )
            questions.append(question)

    if(len(questions) == 0):
        prompt = prepare_prompt(topics, number_of_questions, number_of_answers, sach, bai, chude, mucdo, yccd)
        response = complete_text(prompt)
        new_questions = response_to_questions(response)
        print("new_questions ver 1")
        print(new_questions)
        recent_questions = load_recent_questions(ROOT_PATH + "/exam.json")

        print("recent_questions")
        print(recent_questions)

        
        flagcontinue, contains_num = get_lowest_similarity_exam(new_questions, recent_questions, delta, contains_num)
        if flagcontinue == [] and contains_num != 0:
            get_questions_from_bank(topics, number_of_questions, number_of_answers, sach, bai, chude, mucdo, yccd, contains_num, 0.5)
        else:
            new_ver_ques  = []
            print('else')
            print(new_questions)
            # Create Question objects from the selected data
            for i, (cauhoi, cautraloi, dapan) in enumerate(new_questions):
                # Làm sạch câu hỏi
                cauhoi = sanitize_line(cauhoi, is_question=True)
                # Làm sạch từng đáp án
                answers = [sanitize_line(answer.strip(), is_question=False) for answer in cautraloi.split('~')]

                # Find the most similar answer to the correct answer using PhoBERT embeddings
                correct_answer_index = get_most_similar_answer(dapan, answers)

                # Create the Question object
                question = Question(
                    id=len(questions) + 1,
                    question=cauhoi,
                    answers=answers,
                    correct_answer=correct_answer_index
                )
                new_ver_ques.append(question)
            return new_ver_ques
        print('notelse')
        print(new_questions)
        new_ver_ques  = []
        # Create Question objects from the selected data
        for i, (cauhoi, cautraloi, dapan) in enumerate(new_questions):
            # Làm sạch câu hỏi
            cauhoi = sanitize_line(cauhoi, is_question=True)
            # Làm sạch từng đáp án
            answers = [sanitize_line(answer.strip(), is_question=False) for answer in cautraloi.split('~')]

            # Find the most similar answer to the correct answer using PhoBERT embeddings
            correct_answer_index = get_most_similar_answer(dapan, answers)

            # Create the Question object
            question = Question(
                id=len(questions) + 1,
                question=cauhoi,
                answers=answers,
                correct_answer=correct_answer_index
            )
            new_ver_ques.append(question)
        return new_ver_ques
    elif len(questions) < number_of_questions:
        additional_needed = number_of_questions - len(questions)
        additional_questions = random.sample(filtered_data[['CAUHOI', 'CAUTRALOI', 'DAPAN']].values.tolist(), additional_needed)
        for cauhoi, cautraloi, dapan in additional_questions:
            # Làm sạch câu hỏi
            cauhoi = sanitize_line(cauhoi, is_question=True)  
            # Làm sạch từng đáp án
            answers = [sanitize_line(answer.strip(), is_question=False) for answer in cautraloi.split('~')]
            # Tiếp tục xử lý
            #answers = cautraloi.split('~')
            correct_answer_index = get_most_similar_answer(dapan, answers)
            question = Question(
                id=len(questions) + 1,
                question=cauhoi,
                answers=answers,
                correct_answer=correct_answer_index
            )
            questions.append(question)

    recent_questions = load_recent_questions(
        ROOT_PATH + "/exam.json")
    prompt = prepare_prompt_regenerate(questions)
    response = complete_text(prompt)
    new_questions = response_to_questions(response)

    flagcontinue, contains_num = get_lowest_similarity_exam(new_questions, recent_questions, delta, contains_num)
    if flagcontinue == [] and contains_num != 0:
        get_questions_from_bank(topics, number_of_questions, number_of_answers, sach, bai, chude, mucdo, yccd, contains_num, 0.5)
    else:
        return new_questions
    return new_questions


def load_recent_questions(filename: str) -> List[Question]:
    print('load_recent_questions')

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"File {filename} does not exist.")
        return []  # Return an empty list if the file does not exist

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {filename}: {e}")
        return []  # Return an empty list if there is a JSON error

    # Get current time
    current_time = datetime.now()

    recent_questions = []

    # Filter questions from the last 7 days
    for question_data in data:  # Directly access the questions
        # Check if a timestamp is included, if not use current time
        timestamp_str = question_data.get("timestamp", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            print(f"Error parsing timestamp for question ID {question_data['id']}: {e}")
            continue  # Skip this question if the timestamp is invalid

        if current_time - timestamp <= timedelta(days=7):
            for question in question_data.get("questions"):
                question_info = Question(
                    id=question["id"],
                    question=question["question"].strip(),
                    answers=question["answers"],
                    correct_answer=question["correct_answer"]
                )
                recent_questions.append(question_info)

    return recent_questions


def response_to_questions(response: str) -> List[Question]:
    print('response_to_questions')

    """Convert the response from the API to a list of questions."""
    questions = []
    count = 1

    for question_text in response.split("\n\n"):
        question_text = question_text.strip()
        if not question_text:
            continue

        question_lines = question_text.splitlines()
        if len(question_lines) < 2:  # Đảm bảo câu hỏi có đủ câu trả lời
            continue

        question = sanitize_line(question_lines[0], is_question=True)
        answers = list(map(lambda line: sanitize_line(line, is_question=False), question_lines[1:]))
        correct_answer = get_correct_answer(answers)

        if correct_answer == -1:  # Ensure a correct answer is found
            continue

        answers[correct_answer] = answers[correct_answer].replace("**", "")
        #answers = list(map(lambda answer: answer.strip(), answers))

        questions.append(Question(count, question.strip(), answers, correct_answer))
        count += 1

    return questions


# def calculate_similarity(question1: str, question2: str) -> float:
#     """Calculate the similarity between two questions using multiple methods."""
#     print('calculate_similarity')
#     # Method 1: TF-IDF Cosine Similarity
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform([question1, question2])
#     tfidf_cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
#
#     # Method 2: Word Embeddings
#     def get_word_vector(sentence):
#         words = sentence.split()
#         word_vectors = []
#         for word in words:
#             if word in word2vec_model:
#                 word_vectors.append(word2vec_model[word])
#         return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)  # Adjust dimension as needed
#
#     vector1 = get_word_vector(question1)
#     vector2 = get_word_vector(question2)
#     word_vec_cosine_sim = cosine_similarity([vector1], [vector2])[0][0]
#
#     # Method 3: BERT
#     def get_bert_vector(sentence):
#         inputs = tokenizer_bert(sentence, return_tensors='pt', padding=True, truncation=True)
#         outputs = bert_model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).detach().numpy()
#
#     bert_vector1 = get_bert_vector(question1)
#     bert_vector2 = get_bert_vector(question2)
#     bert_cosine_sim = cosine_similarity(bert_vector1, bert_vector2)[0][0]
#
#     # Combine results
#     final_similarity = (tfidf_cosine_sim + word_vec_cosine_sim + bert_cosine_sim) / 3
#     return final_similarity
def calculate_similarity_batch(new_questions: list, recent_questions: list, delta=0.7, threshold=0.7):
    start_time = time.time()

    # Trích xuất nội dung câu hỏi
    new_question_texts = [q.question for q in new_questions]
    recent_question_texts = [q.question for q in recent_questions]
    try:
        if (recent_question_texts.count() != 0):
            # TF-IDF
            all_questions = new_question_texts + recent_question_texts
            vectorizer = TfidfVectorizer()
            tfidf_vectors = vectorizer.fit_transform(all_questions)
            tfidf_similarities = cosine_similarity(tfidf_vectors[:len(new_question_texts)],
                                                   tfidf_vectors[len(new_question_texts):])
            print("Độ tương đồng TF-IDF:\n", tfidf_similarities)

            # PhoBERT Embeddings
            def get_phobert_vector_batch(sentences):
                inputs = tokenizer_phobert(sentences, return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                    outputs = phobert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).numpy()

            new_phobert_vectors = get_phobert_vector_batch(new_question_texts)
            recent_phobert_vectors = get_phobert_vector_batch(recent_question_texts)
            phobert_similarities = cosine_similarity(new_phobert_vectors, recent_phobert_vectors)
            print("Độ tương đồng PhoBERT:\n", phobert_similarities)

            # SBERT Embeddings
            new_sbert_vectors = sbert_model.encode(new_question_texts, convert_to_tensor=True)
            recent_sbert_vectors = sbert_model.encode(recent_question_texts, convert_to_tensor=True)
            sbert_similarities = cosine_similarity(new_sbert_vectors.cpu(), recent_sbert_vectors.cpu())
            print("Độ tương đồng SBERT:\n", sbert_similarities)

            # Tổng hợp độ tương đồng từ TF-IDF, PhoBERT và SBERT
            final_similarities = (tfidf_similarities + phobert_similarities + sbert_similarities) / 3
            print("Độ tương đồng kết hợp:\n", final_similarities)

            # Tính tỷ lệ số lượng cặp có độ tương đồng nhỏ hơn delta
            num_similar_pairs = np.sum(final_similarities < delta)
            total_pairs = final_similarities.size
            similar_ratio = num_similar_pairs / total_pairs
            print("Tỷ lệ cặp có độ tương đồng nhỏ hơn delta:", similar_ratio)

            # Kiểm tra điều kiện 70%
            result = similar_ratio >= threshold

            end_time = time.time()
            print(f"Tổng thời gian chạy: {end_time - start_time} giây")
        else:
            return new_question_texts
    except:
        return new_question_texts
    return result


def get_lowest_similarity_exam(new_questions: List[Question], recent_questions: List[Question], delta: float,
                               contains_num: int):
    """Find the new questions with the lowest average similarity to existing questions.
       If the average similarity of all new questions exceeds delta, trigger a regeneration."""
    print('get_lowest_similarity_exam')
    if(len(recent_questions) == 0):
        return new_questions, 0
    if calculate_similarity_batch(new_questions, recent_questions, delta, 0.7):
        return new_questions, 0
    else:
        contains_num = contains_num - 1
        return [], contains_num
