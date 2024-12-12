from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class Question:
    """
    Class representing a question

    Attributes:
    - id: Question ID
    - question: Question text
    - answers: List of answers
    - correct_answer: Index of the correct answer
    """
    id: int
    question: str
    answers: List[str]
    correct_answer: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Question object to a dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answers": self.answers,
            "correct_answer": self.correct_answer
        }

# Example usage of saving questions to JSON
def save_questions_to_json(questions: List[Question], filename: str):
    questions_data = [question.to_dict() for question in questions]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"questions": questions_data}, f, ensure_ascii=False, indent=4)
