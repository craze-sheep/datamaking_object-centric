import fitz
import os
import glob

papers_dir = "/home/lzy/project/slot-datamaking/model/research/papers"
pdfs = sorted(glob.glob(os.path.join(papers_dir, "*/*.pdf")))
print(f"Found {len(pdfs)} PDFs")

for pdf_path in pdfs:
    txt_path = pdf_path.replace(".pdf", ".txt")
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 1000:
        print(f"EXISTS: {os.path.relpath(txt_path, papers_dir)}")
        continue
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        size = len(text)
        print(f"OK: {os.path.relpath(txt_path, papers_dir)} ({size} chars)")
    except Exception as e:
        print(f"FAIL: {os.path.relpath(pdf_path, papers_dir)} - {e}")

print("DONE")
