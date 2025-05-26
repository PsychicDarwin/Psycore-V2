from PIL import Image
import io, imagehash,base64, docx2txt,pandas # PyMuPDF
import fitz
from src.system_manager import LoggerController

logger = LoggerController.get_logger()

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
        logger.debug(f"Clustering {len(rects)} rectangles")
        groups = []
        for r in rects:
            for g in groups:
                if any(PDFReader._touches(r, x) for x in g):
                    g.append(r)
                    break
            else:
                groups.append([r])
        logger.debug(f"Created {len(groups)} clusters")
        return groups
    
    def _figure_bbox(page: fitz.Page, draw_rects: list) -> fitz.Rect:
        box = PDFReader._rect_union(draw_rects)
        box = PDFReader._inflate(box, PDFReader.MARGIN_PT)
        
        text_dict = page.get_text("dict")
        blocks_processed = 0
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
                blocks_processed += 1
        logger.debug(f"Processed {blocks_processed} text blocks for bbox calculation")
        return box
    
class FileReader:
    
    def extract_pdf(pdf_path):
        try:
            logger.info(f"Starting PDF extraction for {pdf_path}")
            doc = fitz.open(pdf_path)
            text_content = []
            images = []
            
            processed_hashes = set()
            total_images_found = 0
            total_images_processed = 0

            for page_num, page in enumerate(doc):
                logger.info(f"Processing page {page_num + 1}/{len(doc)}")
                text = page.get_text("text")
                text_content.append(text)

                drawings = page.get_drawings()
                if not drawings:
                    logger.debug(f"No drawings found on page {page_num + 1}")
                    continue
                
                logger.debug(f"Found {len(drawings)} drawings on page {page_num + 1}")
                drects = [fitz.Rect(d["rect"]) for d in drawings]
                clusters = PDFReader._cluster_rects(drects)
                logger.debug(f"Created {len(clusters)} clusters on page {page_num + 1}")

                for cluster_idx, cluster in enumerate(clusters):
                    logger.debug(f"Processing cluster {cluster_idx + 1}/{len(clusters)} on page {page_num + 1}")
                    bbox = PDFReader._figure_bbox(page, cluster)
                    image_list = page.get_images(full=True)
                    total_images_found += len(image_list)
                    logger.debug(f"Found {len(image_list)} images in cluster {cluster_idx + 1}")
                    
                    for img_idx, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Ensure rectangle coordinates are properly ordered
                        x0, y0, x1, y1 = img[1:5]
                        image_rect = fitz.Rect(
                            min(x0, x1),
                            min(y0, y1),
                            max(x0, x1),
                            max(y0, y1)
                        )
                        logger.debug(f"Image {img_idx + 1} rect: {image_rect}, Cluster bbox: {bbox}")
                        if not (image_rect & bbox).is_empty:
                            logger.debug(f"Image {img_idx + 1} intersects with bbox")
                            hash_value = str(imagehash.phash(image))
                            logger.debug(f"Image {img_idx + 1} hash: {hash_value}")
                            logger.debug(f"Current processed hashes: {processed_hashes}")
                            
                            if hash_value not in processed_hashes:
                                logger.debug(f"Found new unique image with hash {hash_value}")
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
                                total_images_processed += 1
                                logger.debug(f"Processed image {img_idx + 1}/{len(image_list)} in cluster {cluster_idx + 1}")
                            else:
                                logger.debug(f"Image {img_idx + 1} was a duplicate (hash already in processed_hashes)")
                
            logger.info(f"PDF processing complete. Found {total_images_found} total images, processed {total_images_processed} unique images")
            return {
                "text": text_content,
                "images": images,
                "page_count": len(doc)
            }
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}", exc_info=True)
            return None
            
    def extract_txt(file_path):
        logger.info(f"Extracting text from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Successfully extracted {len(text)} characters from text file")
        return {"text": text}

    def extract_docx(file_path):
        logger.info(f"Extracting text from {file_path}")
        text = docx2txt.process(file_path)
        logger.info(f"Successfully extracted {len(text)} characters from docx file")
        return {"text": text}
    
    def extract_xlsx(file_path):
        logger.info(f"Extracting data from {file_path}")
        df = pandas.read_excel(file_path)
        text = df.to_string()
        logger.info(f"Successfully extracted {len(df)} rows and {len(df.columns)} columns from xlsx file")
        return {"text": text,
                "rows": len(df),
                "columns": len(df.columns)}
    
    
