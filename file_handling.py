from pypdf import PdfReader

def extract_text(file):
    """This Function Returns list of dictionaries"""
    reader = PdfReader(file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append({
            "text": page.extract_text(),
            "page": page_number
        })

    return pages
