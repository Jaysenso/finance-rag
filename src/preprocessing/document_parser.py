from config import load_config
from src.utils.logger import get_logger
from src.utils.models import load_vision_model

from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem

config = load_config()
logger = get_logger(__name__)

@dataclass
class DocumentElement:
    """Represents a single element from a document."""
    element_id: str
    type: str  # e.g., "Title", "NarrativeText", "Table", "Image"
    text: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_description: Optional[str] = None

@dataclass
class ParsedDocument:
    """Represents a parsed document."""
    file_path: Path
    elements: List[DocumentElement]
    metadata: Dict[str, Any]


class DocumentParser:
    """Document parser using Docling."""

    def __init__(
        self, 
        model: Optional[str] = None,
    ):
        """Initialize document parser."""

        # Configure Docling pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.images_scale = 2.0

        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )
        # Setup vision model for images understanding
        self.vision_client =  load_vision_model(model)

        logger.info(f"Document parser initialized (vision: {model})")

    def _describe_with_vision(self, image: Image.Image, prompt: str) -> str:
        """
        Use vision model to describe an image.
        
        Args:
            image: PIL Image object
            prompt: Prompt for the vision model
            
        Returns:
            Description text
        """
        if not self.vision_client:
            raise ValueError("Vision model not initialized")
        
        return self.vision_client.describe_image(image, prompt)


    def parse_document(
        self,
        file_path: str | Path,
        strategy: str = "auto",
        use_vision_for_tables: bool = True,
        **kwargs,
    ) -> ParsedDocument:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"Parsing document: {file_path.name} (strategy: {strategy})")

        try:
            # Convert document with Docling
            result = self.converter.convert(str(file_path))
            doc = result.document

            # Iterate items and build DocumentElement list
            doc_elements = []
            for idx, (item, level) in enumerate(doc.iterate_items()):
                # Extract page number from provenance
                page_number = None
                if hasattr(item, 'prov') and item.prov:
                    page_refs = [p.page_no for p in item.prov if hasattr(p, 'page_no')]
                    page_number = page_refs[0] if page_refs else None

                if isinstance(item, TableItem):
                    table_md = item.export_to_markdown(doc)

                    # Optionally use vision model to refine the table markdown
                    if use_vision_for_tables and self.vision_client:
                        try:
                            table_image = item.get_image(doc)
                            if table_image:
                                table_md = self._describe_with_vision(
                                    table_image,
                                    "Below is a markdown table extracted via OCR from a "
                                    "financial document. Compare it against the table image "
                                    "and correct any OCR errors, missing values, or "
                                    "formatting issues. Return ONLY the corrected markdown "
                                    "table, nothing else.\n\n"
                                    f"{table_md}",
                                )
                                print(table_md)
                        except Exception as e:
                            logger.warning(f"Vision refinement failed for table, using raw markdown: {e}")
                   
                    doc_elements.append(DocumentElement(
                        element_id=f"elem_{idx}",
                        type="Table",
                        text=table_md,
                        page_number=page_number,
                    ))

                elif isinstance(item, PictureItem):
                    # Use vision model to describe the image
                    image_description = None
                    try:
                        picture_image = item.get_image(doc)
                        if picture_image and self.vision_client:
                            image_description = self._describe_with_vision(
                                picture_image,
                                "Describe this image from a financial document. "
                                "What does it show? Include key details and data.",
                            )
                    except Exception as e:
                        logger.warning(f"Could not get vision description for image: {e}")

                    doc_elements.append(DocumentElement(
                        element_id=f"elem_{idx}",
                        type="Image",
                        text=image_description or "[Image]",
                        page_number=page_number,
                        metadata={},
                        image_description=image_description,
                    ))

                else:
                    # Text element - map Docling label to our type
                    text = item.text if hasattr(item, 'text') else str(item)
                    label = getattr(item, 'label', None)
                    label_str = label.value if hasattr(label, 'value') else str(label).lower()

                    if label_str in ('title', 'section_header'):
                        elem_type = "Title"
                    elif label_str == 'list_item':
                        elem_type = "ListItem"
                    else:
                        elem_type = "NarrativeText"

                    doc_elements.append(DocumentElement(
                        element_id=f"elem_{idx}",
                        type=elem_type,
                        text=text,
                        page_number=page_number,
                        metadata={},
                    ))

            metadata = {
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "num_elements": len(doc_elements),
                "num_tables": len([e for e in doc_elements if e.type == "Table"]),
                "num_images": len([e for e in doc_elements if e.type == "Image"]),
            }

            parsed_doc = ParsedDocument(
                file_path=file_path,
                elements=doc_elements,
                metadata=metadata,
            )

            logger.info(
                f"Parsed {len(doc_elements)} elements "
                f"({metadata['num_tables']} tables, {metadata['num_images']} images)"
            )

            return parsed_doc

        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {e}")
            raise

if __name__ == "__main__":

    # # Test document parser
    parser = DocumentParser()
    file_path = "./src/data/pdf/sample/sample-unstructured-paper.pdf"
    
    parsed_docs = parser.parse_document(file_path)
    print(parsed_docs)
