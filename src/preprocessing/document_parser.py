from config import load_config
from src.utils.logger import get_logger
from src.utils.models import load_vision_model
from src.preprocessing.models import DocumentElement, ParsedDocument

from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem

try:
    import pandas as pd
except ImportError:
    pd = None
    
try:
    import openpyxl
except ImportError:
    openpyxl = None

config = load_config()
logger = get_logger(__name__)


class DocumentParser:
    """Document parser using Docling."""

    def __init__(
        self, 
        model: Optional[str] = None,
        use_vision_for_tables: bool = False,
        max_vision_workers: int = 4,
    ):
        """Initialize document parser.
        
        Args:
            model: Vision model to use for image/table processing
            use_vision_for_tables: Whether to use vision model for tables (default: True)
            max_vision_workers: Max parallel workers for vision processing (default: 4)
        """

        # Configure Docling pipeline options
        accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA
        )
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
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
        # ALWAYS load vision client if model is provided, regardless of table setting
        self.use_vision_for_tables = use_vision_for_tables
        self.max_vision_workers = max_vision_workers
        self.vision_client = load_vision_model(model) if model else None

        logger.info(
            f"Document parser initialized (vision_model: {'loaded' if self.vision_client else 'disabled'}, "
            f"table_vision: {use_vision_for_tables}, workers: {max_vision_workers})"
        )

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

    def _process_vision_items(
        self, 
        vision_tasks: List[tuple[int, Any, Image.Image, str]]
    ) -> Dict[int, str]:
        """
        Process multiple vision tasks in parallel.
        
        Args:
            vision_tasks: List of (index, item, image, prompt) tuples
            
        Returns:
            Dictionary mapping index to vision description
        """
        results = {}
        
        if not vision_tasks or not self.vision_client:
            return results
        
        def process_single_vision(task):
            idx, item, image, prompt = task
            try:
                description = self._describe_with_vision(image, prompt)
                return idx, description
            except Exception as e:
                logger.warning(f"Vision processing failed for item {idx}: {e}")
                return idx, None
        
        # Process vision tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_vision_workers) as executor:
            futures = {executor.submit(process_single_vision, task): task for task in vision_tasks}
            
            for future in as_completed(futures):
                idx, description = future.result()
                if description:
                    results[idx] = description
        
        return results

    def _parse_excel(self, file_path: Path) -> List[DocumentElement]:
        """Parse Excel file into table elements."""
        if pd is None or openpyxl is None:
            raise ImportError("pandas and openpyxl are required for Excel parsing")
            
        elements = []
        try:
            # Read all sheets
            xls = pd.ExcelFile(file_path)
            for i, sheet_name in enumerate(xls.sheet_names):
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # Convert to markdown
                markdown_table = df.to_markdown(index=False)
                
                elements.append(DocumentElement(
                    element_id=f"sheet_{i}",
                    type="Table",
                    text=markdown_table,
                    page_number=i+1, # Treat sheets as pages
                    metadata={"sheet_name": sheet_name}
                ))
            return elements
        except Exception as e:
            logger.error(f"Failed to parse Excel file {file_path}: {e}")
            raise

    def _parse_image(self, file_path: Path) -> List[DocumentElement]:
        """Parse Image file using vision model."""
        if not self.vision_client:
            logger.warning(f"Vision model disabled/not configured, skipping image: {file_path}")
            return []
            
        try:
            image = Image.open(file_path)
            prompt = (
                "Analyze this image from a financial context. "
                "Describe any charts, graphs, or data tables visible. "
                "Extract key figures and trends."
            )
            description = self._describe_with_vision(image, prompt)
            
            return [DocumentElement(
                element_id="img_0",
                type="Image",
                text=description,
                page_number=1,
                metadata={},
                image_description=description
            )]
        except Exception as e:
            logger.error(f"Failed to parse image file {file_path}: {e}")
            raise

    def parse_documents_batch(
        self,
        file_paths: List[str | Path],
        use_vision_for_tables: Optional[bool] = None,
        **kwargs,
    ) -> List[ParsedDocument]:
        """
        Parse multiple documents in batch for improved performance.
        
        Args:
            file_paths: List of file paths to parse
            strategy: Parsing strategy (default: "auto")
            use_vision_for_tables: Override table vision setting (default: None, use instance setting)
            
        Returns:
            List of ParsedDocument objects
        """
        # Resolve table vision setting
        vision_tables = use_vision_for_tables if use_vision_for_tables is not None else self.use_vision_for_tables
        file_paths = [Path(fp) for fp in file_paths]
        
        # Validate all files exist
        for fp in file_paths:
            if not fp.exists():
                raise FileNotFoundError(f"Document not found: {fp}")
        
        logger.info(f"Batch parsing {len(file_paths)} documents")
        
        try:
            # Separate Docling-supported files from others
            docling_files = []
            other_docs = [] # List of ParsedDocument

            
            for fp in file_paths:
                suffix = fp.suffix.lower()

                # PDF/Docx
                if suffix in ['.pdf', '.docx', '.html']:
                    docling_files.append(str(fp))

                elif suffix in ['.xlsx', '.xls']:
                    logger.info(f"Parsing Excel: {fp.name}")
                    elements = self._parse_excel(fp)
                    other_docs.append(ParsedDocument(
                        file_path=fp,
                        elements=elements,
                        metadata={
                            "file_name": fp.name,
                            "file_type": suffix,
                            "num_elements": len(elements),
                            "num_tables": len(elements),
                            "num_images": 0
                        }
                    ))

                # Image
                elif suffix in ['.jpg', '.jpeg', '.png']:
                    logger.info(f"Parsing Image: {fp.name}")
                    elements = self._parse_image(fp)
                    other_docs.append(ParsedDocument(
                        file_path=fp,
                        elements=elements,
                        metadata={
                            "file_name": fp.name,
                            "file_type": suffix,
                            "num_elements": len(elements),
                            "num_tables": 0,
                            "num_images": len(elements)
                        }
                    ))
                else:
                    logger.warning(f"Unsupported file type: {fp}")

            # Batch convert Docling-supported documents
            results = []
            if docling_files:
                results = self.converter.convert_all(docling_files)
            
            # Collect all vision tasks across all documents
            all_vision_tasks = []
            all_items_data = [] # For docling docs only
            
            # Map docling results back to file paths 
            # docling_files contains the paths passed to convert_all
            for doc_idx, (result, file_path_str) in enumerate(zip(results, docling_files)):
                file_path = Path(file_path_str)
                doc = result.document
                items_data = []
                
                for idx, (item, level) in enumerate(doc.iterate_items()):
                    # Extract page number from provenance
                    page_number = None

                    if hasattr(item, 'prov') and item.prov:
                        page_refs = [p.page_no for p in item.prov if hasattr(p, 'page_no')]
                        page_number = page_refs[0] if page_refs else None
                    
                    items_data.append((idx, item, level, page_number))
                    
                    # Collect vision tasks for parallel processing
                    if self.vision_client:
                        if isinstance(item, TableItem) and vision_tables:
                            try:
                                table_image = item.get_image(doc)
                                table_md = item.export_to_markdown(doc)

                                if table_image:
                                    prompt = (
                                        "Below is a markdown table extracted via OCR from a "
                                        "financial document. Compare it against the table image "
                                        "and correct any OCR errors, missing values, or "
                                        "formatting issues. Return ONLY the corrected markdown "
                                        f"table, nothing else.\n\n{table_md}"
                                    )
                                    all_vision_tasks.append(((doc_idx, idx), item, table_image, prompt))
                            except Exception as e:
                                logger.warning(f"Could not prepare table vision task: {e}")
                        
                        elif isinstance(item, PictureItem):
                            try:
                                picture_image = item.get_image(doc)
                                if picture_image:
                                    prompt = (
                                        "Describe this image from a financial document. "
                                        "What does it show? Include key details and data."
                                    )
                                    all_vision_tasks.append(((doc_idx, idx), item, picture_image, prompt))
                            except Exception as e:
                                logger.warning(f"Could not prepare image vision task: {e}")
                
                all_items_data.append((doc, items_data, file_path))
            
            # Process all vision tasks in parallel across all documents
            vision_results = self._process_vision_items(all_vision_tasks)
            
            # Build ParsedDocument objects
            parsed_docs = []
            for doc_idx, (doc, items_data, file_path) in enumerate(all_items_data):
                doc_elements = []
                
                for idx, item, level, page_number in items_data:
                    composite_key = (doc_idx, idx)
                    
                    if isinstance(item, TableItem):
                        table_md = vision_results.get(composite_key) or item.export_to_markdown(doc)
                        
                        doc_elements.append(DocumentElement(
                            element_id=f"elem_{idx}",
                            type="Table",
                            text=table_md,
                            page_number=page_number,
                        ))
                    
                    elif isinstance(item, PictureItem):
                        image_description = vision_results.get(composite_key)
                        
                        doc_elements.append(DocumentElement(
                            element_id=f"elem_{idx}",
                            type="Image",
                            text=image_description or "[Image]",
                            page_number=page_number,
                            metadata={},
                            image_description=image_description,
                        ))
                    
                    else:
                        # Text element
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
                
                parsed_docs.append(ParsedDocument(
                    file_path=file_path,
                    elements=doc_elements,
                    metadata=metadata,
                ))
                
                logger.info(
                    f"Parsed {file_path.name}: {len(doc_elements)} elements "
                    f"({metadata['num_tables']} tables, {metadata['num_images']} images)"
                )
            
            logger.info(f"Batch parsing complete: {len(parsed_docs)} documents processed")
            
            # Combine docling docs with other docs
            return parsed_docs + other_docs
        
        except Exception as e:
            logger.error(f"Error in batch parsing: {e}")
            raise


def get_document_parser(**kwargs) -> DocumentParser:
    """
    Function to create a DocumentParser instance.

    Returns:
        DocumentParser instance
    """
    doc_config = config.get("document_parsing", {})
    vision_llm_config = config.get("vision_llm", {})
    
    # Set defaults from config
    defaults = {
        "model": vision_llm_config.get("model", None),
        "use_vision_for_tables": doc_config.get("use_vision_for_tables", True), 
        "max_vision_workers": doc_config.get("max_vision_workers", 4),
    }
    
    defaults.update(kwargs)
    logger.info("Creating document parser")
    return DocumentParser(**defaults)


if __name__ == "__main__":

    # # Test document parser
    parser = get_document_parser()
    file_path = "./src/data/pdf/sample/sample-unstructured-paper.pdf"
    
    parsed_docs = parser.parse_documents_batch([file_path])
    print(parsed_docs)

