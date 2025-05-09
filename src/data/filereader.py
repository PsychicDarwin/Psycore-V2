from PIL import Image
import io, imagehash,base64, docx2txt,pandas # PyMuPDF
import fitz
class PDFReader:
    
    MARGIN_PT = 3.0
    LABEL_PAD = 18.0
    MAX_CHAR = 350
    MIN_OBJECTS = 15
    def _inflate(rect: fitz.Rect, d: float) -> fitz.Rect:
        return fitz.Rect(
            rect.x0 - d,
            rect.y0 - d,
            rect.x1 + d,
            rect.y1 + d
        )
    
    def _rect_union(rects: list[fitz.Rect]) -> fitz.Rect:
        r0 = rects[0]
        for r in rects[1:]:
            r0 = r0 | r
        return r0
    
    def _touches(a: fitz.Rect, b: fitz.Rect, gap :float = 0.7) -> bool:
        return bool(PDFReader._inflate(a,gap) & b)
    
    def _cluster_rects(rects: list) -> list:
        groups = []
        for r in rects:
            for g in groups:
                if any(PDFReader._touches(r, x) for x in g):
                    g.append(r)
                    break
            else:
                groups.append([r])
        return groups
    
    def _figure_bbox(page: fitz.Page, draw_rects: list) -> fitz.Rect:
        box = PDFReader._rect_union(draw_rects)
        box = PDFReader._inflate(box, PDFReader.MARGIN_PT)
        
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if block["type"] != 0:
                continue
            chars = sum(len(span["text"]) for line in block["lines"] for span in line["spans"])
            if chars > PDFReader.MAX_CHAR:
                continue
            rect = fitz.Rect(block["bbox"])
            near = PDFReader._inflate(box, PDFReader.LABEL_PAD)
            if near & rect:
                box = box | rect
        return box
    
class FileReader:
    
    def extract_pdf(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            images = []
            
            processed_hashes = set()

            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                text_content.append(text)

                drawings = page.get_drawings()
                if not drawings:
                    continue
                drects = [fitz.Rect(d["rect"]) for d in drawings]
                clusters = PDFReader._cluster_rects(drects)

                for cluster in clusters:
                    bbox = PDFReader._figure_bbox(page, cluster)
                    image_list = page.get_images(full=True, clip=bbox)
                    for img in image_list:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))
                        hash_value = str(imagehash.phash(image))
                        
                        if hash_value not in processed_hashes:
                            processed_hashes.add(hash_value)
                            buffer = io.BytesIO()
                            image.convert("RGB").save(buffer, format="JPEG")
                            jpeg_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                            images.append({
                                "page": page_num,
                                "bbox": bbox,
                                "image": jpeg_data,
                                "surrounding_text": text
                            })
                return {
                    "text": text_content,
                    "images": images,
                    "page_count": len(doc)
                }
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return None
            
    def extract_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return {"text": text}

    def extract_docx(file_path):
        text = docx2txt.process(file_path)
        return {"text": text}
    
    def extract_xlsx(file_path):
        df = pandas.read_excel(file_path)
        text = df.to_string()
        return {"text": text,
                "rows": len(df),
                "columns": len(df.columns)}
    
    
