# iterable: Any object that supports iteration
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path) :
    doc = fitz.open(pdf_path)
    # open pdf
    text = ""
    for page in doc :
        text += page.get_text() 
        # iterate through each page and extract text

    return text

print(extract_text_from_pdf("sample.pdf"))