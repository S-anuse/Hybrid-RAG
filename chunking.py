# chunk size tells us how much text goes into each chunk.
# overlap : it helps preserve information around boundaries.

import fitz  # PyMuPDF

def chunk_text(pdf_path , chunk_size , overlap) :
    chunks = []

    doc = fitz.open(pdf_path)

    paragraphs = []
    
    for page in doc :
        page_text = page.get_text()

        page_paragraphs = page_text.split("\n\n")
        
        for paragraph in page_paragraphs :
            paragraph = paragraph.strip()
            if paragraph :
                paragraphs.append({
                    "text" : paragraph , 
                    "page" : page.number + 1    
                })
        
    # create chunks from paragraphs
    id = 1
    curr_chunk = ""
    chunk_page = set()
    for paragraph in paragraphs :
        if len(curr_chunk) + len(paragraph["text"]) + 1 <= chunk_size :
            curr_chunk += paragraph["text"] + "\n"
            chunk_page.add(paragraph["page"])
        else :
            if curr_chunk :
                chunks.append([curr_chunk.strip() ,pdf_path , id , chunk_page])
            id += 1
            curr_chunk = paragraph["text"] + "\n"
            chunk_page = {paragraph["page"]}

    if curr_chunk :
        chunks.append([curr_chunk.strip() , pdf_path , id , chunk_page])

    return chunks


for idx , t in enumerate(chunk_text("sample.pdf", 1000, 200)) :
    print(f"Chunk {idx+1} : {t}\n\n")