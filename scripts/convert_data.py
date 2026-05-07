"""
Utility script to convert data formats for pretraining.
Supports converting from various formats to JSONL.
Handles single files or entire directories with mixed file types.
"""

import argparse
import json
import logging
import csv
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional dependencies for different formats
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed - Parquet/Excel support disabled")

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    logger.warning("PyPDF2 not installed - PDF support disabled")

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    logger.warning("pyarrow not installed - Parquet support disabled")


class UniversalConverter:
    """Convert various document formats to JSONL."""
    
    def __init__(self, output_file: str, text_column: str = "text"):
        self.output_file = Path(output_file)
        self.text_column = text_column
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.total_documents = 0
    
    def convert_directory(
        self,
        directory: str,
        recursive: bool = True,
        max_workers: int = 4,
    ):
        """
        Convert all files in directory to JSONL.
        
        Args:
            directory: Path to directory with documents
            recursive: Scan subdirectories
            max_workers: Number of parallel workers
        """
        dir_path = Path(directory)
        
        # Find all supported files
        files_to_convert = self._find_supported_files(dir_path, recursive)
        
        if not files_to_convert:
            logger.warning(f"No supported files found in {directory}")
            return
        
        logger.info(f"Found {len(files_to_convert)} files to convert")
        
        # Open output file for writing (truncate if exists)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            pass
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_file, file_path): file_path
                for file_path in files_to_convert
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    texts = future.result()
                    if texts:
                        self._append_to_jsonl(texts)
                        logger.info(f"✓ Processed: {file_path.name} ({len(texts)} docs)")
                except Exception as e:
                    logger.error(f"✗ Failed to process {file_path}: {e}")
        
        logger.info(f"✓ Complete! Total documents: {self.total_documents}")
    
    def _find_supported_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all supported file types."""
        supported_extensions = {
            '.txt', '.csv', '.json',
            '.parquet', '.pq',
            '.xlsx', '.xls',
        }
        
        if HAS_PDF:
            supported_extensions.add('.pdf')
        
        pattern = '**/*' if recursive else '*'
        files = []
        
        for ext in supported_extensions:
            files.extend(directory.glob(f"{pattern}{ext}"))
        
        return sorted(set(files))  # Remove duplicates
    
    def _process_file(self, file_path: Path) -> List[str]:
        """Process single file based on type."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                return self._read_txt(file_path)
            elif suffix == '.csv':
                return self._read_csv(file_path)
            elif suffix == '.json':
                return self._read_json(file_path)
            elif suffix in ['.parquet', '.pq']:
                return self._read_parquet(file_path)
            elif suffix in ['.xlsx', '.xls']:
                return self._read_excel(file_path)
            elif suffix == '.pdf':
                return self._read_pdf(file_path)
            else:
                logger.warning(f"Unsupported format: {suffix}")
                return []
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def _read_txt(self, file_path: Path) -> List[str]:
        """Read plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
        
        return [text] if text else []
    
    def _read_csv(self, file_path: Path) -> List[str]:
        """Read CSV file."""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.text_column in row:
                        text = row[self.text_column].strip()
                        if text:
                            texts.append(text)
        except Exception as e:
            logger.warning(f"Could not read {file_path} as CSV: {e}")
        
        return texts
    
    def _read_json(self, file_path: Path) -> List[str]:
        """Read JSON file (JSONL or JSON array)."""
        texts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try JSONL format (one JSON per line)
        try:
            for line in content.split('\n'):
                if line.strip():
                    data = json.loads(line)
                    if isinstance(data, dict):
                        text = data.get(self.text_column, '').strip()
                        if text:
                            texts.append(text)
                    elif isinstance(data, str):
                        texts.append(data)
            return texts
        except json.JSONDecodeError:
            pass
        
        # Try JSON array format
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = item.get(self.text_column, '').strip()
                        if text:
                            texts.append(text)
                    elif isinstance(item, str):
                        texts.append(item)
            elif isinstance(data, dict):
                text = data.get(self.text_column, '').strip()
                if text:
                    texts.append(text)
            return texts
        except json.JSONDecodeError:
            pass
        
        return texts
    
    def _read_parquet(self, file_path: Path) -> List[str]:
        """Read Parquet file."""
        if not HAS_PARQUET:
            logger.warning("pyarrow not installed - skipping Parquet file")
            return []
        
        texts = []
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            if self.text_column in df.columns:
                for text in df[self.text_column]:
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
        except Exception as e:
            logger.warning(f"Could not read Parquet {file_path}: {e}")
        
        return texts
    
    def _read_excel(self, file_path: Path) -> List[str]:
        """Read Excel file (.xlsx, .xls)."""
        if not HAS_PANDAS:
            logger.warning("pandas not installed - skipping Excel file")
            return []
        
        texts = []
        try:
            df = pd.read_excel(file_path)
            
            if self.text_column in df.columns:
                for text in df[self.text_column]:
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
            elif len(df.columns) > 0:
                # If text_column doesn't exist, use first column
                col = df.columns[0]
                for text in df[col]:
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
        except Exception as e:
            logger.warning(f"Could not read Excel {file_path}: {e}")
        
        return texts
    
    def _read_pdf(self, file_path: Path) -> List[str]:
        """Extract text from PDF."""
        if not HAS_PDF:
            logger.warning("PyPDF2 not installed - skipping PDF files")
            return []
        
        texts = []
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text().strip()
                    if text:
                        texts.append(text)
        except Exception as e:
            logger.warning(f"Could not extract text from PDF {file_path}: {e}")
        
        return texts
    
    def _append_to_jsonl(self, texts: List[str]):
        """Append texts to JSONL file."""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for text in texts:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write('\n')
                self.total_documents += 1


# Legacy functions for backward compatibility
def convert_txt_to_jsonl(input_file: str, output_file: str, chunk_size: int = None):
    """Convert single text file to JSONL."""
    logger.info(f"Converting {input_file} to JSONL format")
    converter = UniversalConverter(output_file)
    texts = converter._read_txt(Path(input_file))
    converter._append_to_jsonl(texts)


def convert_csv_to_jsonl(
    input_file: str,
    output_file: str,
    text_column: str = "text",
    delimiter: str = ",",
):
    """Convert single CSV file to JSONL."""
    logger.info(f"Converting CSV {input_file} to JSONL format")
    converter = UniversalConverter(output_file, text_column=text_column)
    texts = converter._read_csv(Path(input_file))
    converter._append_to_jsonl(texts)


def split_jsonl(input_file: str, output_dir: str, train_ratio: float = 0.9):
    """Split JSONL file into train/val sets."""
    logger.info(f"Splitting {input_file} (train_ratio={train_ratio})")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    split_point = int(len(lines) * train_ratio)
    train_lines = lines[:split_point]
    val_lines = lines[split_point:]
    
    train_file = Path(output_dir) / "train.jsonl"
    val_file = Path(output_dir) / "val.jsonl"
    
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
    
    with open(val_file, 'w') as f:
        f.writelines(val_lines)
    
    logger.info(f"Train: {train_file} ({len(train_lines)} lines)")
    logger.info(f"Val: {val_file} ({len(val_lines)} lines)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert various data formats to JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_data.py --input file.txt --output output.jsonl --format txt
  
  # Convert entire directory with mixed formats
  python convert_data.py --input ./data --output data.jsonl --mode directory
  
  # Convert with custom column name
  python convert_data.py --input data.csv --output output.jsonl --format csv --text_column "content"
  
  # Split into train/val
  python convert_data.py --input output.jsonl --output ./processed --format jsonl --split 0.9
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path or directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "csv", "json", "parquet", "excel", "directory"],
        default="txt",
        help="Input format (or 'directory' for mixed types)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name for text (CSV/Parquet/Excel)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="Train/val split ratio (0-1)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers for directory processing",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively process subdirectories",
    )
    
    args = parser.parse_args()
    
    # Handle directory conversion
    if args.format == "directory" or Path(args.input).is_dir():
        converter = UniversalConverter(args.output, text_column=args.text_column)
        converter.convert_directory(
            args.input,
            recursive=args.recursive,
            max_workers=args.max_workers,
        )
    else:
        # Single file conversion
        if args.format == "txt":
            convert_txt_to_jsonl(args.input, args.output)
        elif args.format == "csv":
            convert_csv_to_jsonl(args.input, args.output, text_column=args.text_column)
        elif args.format == "json":
            converter = UniversalConverter(args.output)
            texts = converter._read_json(Path(args.input))
            converter._append_to_jsonl(texts)
        elif args.format == "parquet":
            converter = UniversalConverter(args.output, text_column=args.text_column)
            texts = converter._read_parquet(Path(args.input))
            converter._append_to_jsonl(texts)
        elif args.format == "excel":
            converter = UniversalConverter(args.output, text_column=args.text_column)
            texts = converter._read_excel(Path(args.input))
            converter._append_to_jsonl(texts)
    
    # Split if requested
    if args.split:
        output_dir = Path(args.output).parent
        split_jsonl(args.output, str(output_dir), train_ratio=args.split)