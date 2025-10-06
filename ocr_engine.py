from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# model = ocr_predictor(pretrained=True)
model = ocr_predictor("fast_mini", pretrained=True)

def process_doc(tmp_path, filename):
    if filename.lower().endswith(".pdf"):
        doc = DocumentFile.from_pdf(tmp_path)
    else:
        doc = DocumentFile.from_images(tmp_path)
    return doc

def structure_doctr_result(result):
    """
    Structure DocTR OCR result into readable text.
    Sorts words by position and groups into lines with proper spacing.

    Args:
        result: DocTR model prediction result

    Returns:
        Structured text as string
    """

    structured_lines = []

    for page in result.pages:
        page_height = page.dimensions[1]
        page_width = page.dimensions[0]

        all_words = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geom = word.geometry
                    x_left = geom[0][0] * page_width
                    x_right = geom[1][0] * page_width
                    x_center = (x_left + x_right) / 2
                    y_center = (geom[0][1] + geom[1][1]) / 2 * page_height
                    word_width = x_right - x_left

                    all_words.append({
                        'text': word.value,
                        'x': x_center,
                        'x_left': x_left,
                        'x_right': x_right,
                        'y': y_center,
                        'width': word_width,
                        'confidence': word.confidence
                    })

        all_words.sort(key=lambda w: (w['y'], w['x']))

        lines = []
        current_line = []
        current_y = -1
        y_threshold = 15

        for word in all_words:
            if current_y == -1 or abs(word['y'] - current_y) <= y_threshold:
                current_line.append(word)
                if current_y == -1:
                    current_y = word['y']
                else:
                    current_y = (current_y + word['y']) / 2
            else:
                current_line.sort(key=lambda w: w['x'])
                line_text = format_line_with_spacing(current_line, page_width)
                lines.append(line_text)

                current_line = [word]
                current_y = word['y']

        if current_line:
            current_line.sort(key=lambda w: w['x'])
            line_text = format_line_with_spacing(current_line, page_width)
            lines.append(line_text)

        structured_lines.extend(lines)

    return '\n'.join(structured_lines)


def format_line_with_spacing(words, page_width):
    """
    Format a line of words with proper spacing based on horizontal gaps.

    Args:
        words: List of word dictionaries sorted by x position
        page_width: Width of the page in pixels

    Returns:
        Formatted line with appropriate spacing
    """
    if not words:
        return ""

    if len(words) == 1:
        return words[0]['text']

    avg_word_width = sum(w['width'] for w in words) / len(words)

    result = [words[0]['text']]

    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]
        gap = curr_word['x_left'] - prev_word['x_right']

        if gap > avg_word_width * 3:
            result.append('    ')
        elif gap > avg_word_width * 1.5:
            result.append('  ')
        else:
            result.append(' ')

        result.append(curr_word['text'])

    return ''.join(result)