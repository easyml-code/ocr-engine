from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from doctr.io import DocumentFile
from ocr_engine import structure_doctr_result, model, process_doc
import tempfile
import shutil
import os

app = FastAPI(title="OCR API", version="0.0.1")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running fine!"}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """
    Accepts an image or PDF file and returns extracted text.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        try: 
            doc = process_doc(tmp_path, file.filename)
        except Exception as e:
            os.remove(tmp_path)
            return JSONResponse(status_code=400, content={"error": "Unsupported file format or corrupted file."})

        result = model(doc)
        structured_text = structure_doctr_result(result)

        os.remove(tmp_path)

        return JSONResponse(status_code=200, content={"filename": file.filename, "extracted_text": structured_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})