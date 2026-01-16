from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from google.cloud.vision_v1 import types
from src.utils.logging import get_logger

logger = get_logger("region_processor_logs")

@dataclass
class Region:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class AnswerScriptProcessor:
    def __init__(self):
        # Adjusted region values based on the sample image
        self.question_box_region = Region(
            x_min=0.0,    # Leftmost of page
            y_min=0.0,    # Top of page
            x_max=0.1,    # Reduced to 10% from left to avoid capturing extra characters
            y_max=0.2     # Reduced to focus on just the question number area
        )
        
        self.answer_region = Region(
            x_min=0.1,    # After question box
            y_min=0.0,    # Top of page
            x_max=1.0,    # Right edge
            y_max=1.0     # Bottom of page
        )

    def clean_question_id(self, text: str) -> str:
        """Clean and extract just the numeric question ID."""
        # Remove any non-numeric characters
        numeric = ''.join(char for char in text if char.isdigit())
        return numeric if numeric else ""

    def is_in_region(self, vertices: List[types.Vertex], region: Region, image_width: int, image_height: int) -> bool:
        """Check if text block is within specified region."""
        x_coords = [v.x / image_width for v in vertices]
        y_coords = [v.y / image_height for v in vertices]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return (region.x_min <= center_x <= region.x_max and
                region.y_min <= center_y <= region.y_max)

    def extract_regions(self, text_annotations, image_width: int, image_height: int) -> Tuple[str, str]:
        """Extract and clean text from question ID and answer regions separately."""
        question_id_texts = []
        answer_texts = []

        # Create a mapping of y-positions to text for proper ordering
        for annotation in text_annotations[1:]:
            vertices = annotation.bounding_poly.vertices
            text = annotation.description
            center_y = sum(v.y for v in vertices) / 4

            if self.is_in_region(vertices, self.question_box_region, image_width, image_height):
                question_id_texts.append((text, center_y))
            elif self.is_in_region(vertices, self.answer_region, image_width, image_height):
                answer_texts.append((text, center_y))

        # Sort by vertical position and join texts
        question_id_texts.sort(key=lambda x: x[1])
        answer_texts.sort(key=lambda x: x[1])

        question_id = self.clean_question_id(' '.join(text for text, _ in question_id_texts))
        answer_text = ' '.join(text for text, _ in answer_texts)

        return question_id, answer_text