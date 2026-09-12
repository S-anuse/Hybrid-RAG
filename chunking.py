# chunk size tells us how much text goes into each chunk.
# overlap : it helps preserve information around boundaries.

import fitz  # PyMuPDF

def chunk_texts(pdf_path , chunk_size , overlap) :
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
    curr_paragraphs = []
    chunk_page = set()
    for paragraph in paragraphs :
        if len(curr_chunk) + len(paragraph["text"]) + 1 <= chunk_size :
            curr_chunk += paragraph["text"] + "\n"
            chunk_page.add(paragraph["page"])
            curr_paragraphs.append(paragraph)
        else :
            if curr_chunk :
                chunks.append([curr_chunk.strip() ,pdf_path , id , chunk_page])
            id += 1
            curr_chunk = ""
            chunk_page = set()
            curr_paragraphs = curr_paragraphs[-overlap:]
            for p in curr_paragraphs :
                curr_chunk += p["text"] + "\n"
                chunk_page.add(p["page"])
            curr_chunk += paragraph["text"] + "\n"
            chunk_page.add(paragraph["page"])
            curr_paragraphs.append(paragraph)

    if curr_chunk :
        chunks.append([curr_chunk.strip() , pdf_path , id , chunk_page])

    return chunks


if __name__ == "__main__":
    for idx, t in enumerate(chunk_texts("sample.pdf", 1000, 2)):
        print(f"Chunk {idx+1} : {t}\n\n")
