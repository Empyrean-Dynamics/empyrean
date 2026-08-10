//! Parquet row source — one record batch resident at a time.

use std::path::{Path, PathBuf};

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{
    Float16Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};

use super::{Cell, RowSource, ShowError};

/// Rows decoded per pull from the Parquet reader.
///
/// This bounds resident memory: an 82-column orbits table costs roughly
/// `BATCH_ROWS × 82` cells regardless of the file's row count. It is set
/// well above a screenful so paging forward does not re-enter the
/// decoder on every keypress, and well below a row group so a wide file
/// never materialises one whole group.
const BATCH_ROWS: usize = 1024;

pub struct ParquetSource {
    path: PathBuf,
    columns: Vec<String>,
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    /// The batch currently being handed out, and how far into it we are.
    batch: Option<RecordBatch>,
    row_in_batch: usize,
}

impl ParquetSource {
    pub fn open(path: &Path) -> Result<Self, ShowError> {
        let file = std::fs::File::open(path).map_err(|e| ShowError::io(path, "open", e))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| ShowError::parquet(path, "read the parquet footer of", e))?;
        let columns = builder
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let reader = builder
            .with_batch_size(BATCH_ROWS)
            .build()
            .map_err(|e| ShowError::parquet(path, "open a record-batch reader over", e))?;
        Ok(Self {
            path: path.to_path_buf(),
            columns,
            reader,
            batch: None,
            row_in_batch: 0,
        })
    }
}

impl RowSource for ParquetSource {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Vec<Cell>>, ShowError> {
        loop {
            if let Some(batch) = &self.batch {
                if self.row_in_batch < batch.num_rows() {
                    let row = (0..batch.num_columns())
                        .map(|c| cell_at(&self.path, &self.columns[c], batch, c, self.row_in_batch))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.row_in_batch += 1;
                    return Ok(Some(row));
                }
                self.batch = None;
            }
            match self.reader.next() {
                None => return Ok(None),
                Some(Err(e)) => {
                    return Err(ShowError::parquet(
                        &self.path,
                        "decode a record batch from",
                        parquet::errors::ParquetError::ArrowError(e.to_string()),
                    ));
                }
                Some(Ok(batch)) => {
                    self.batch = Some(batch);
                    self.row_in_batch = 0;
                }
            }
        }
    }
}

/// Exact row count from the Parquet footer — no page is decoded.
pub fn row_count(path: &Path) -> Result<u64, ShowError> {
    let file = std::fs::File::open(path).map_err(|e| ShowError::io(path, "open", e))?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| ShowError::parquet(path, "read the parquet footer of", e))?;
    Ok(reader.metadata().file_metadata().num_rows().max(0) as u64)
}

/// Convert one Arrow value to a [`Cell`].
///
/// The match is deliberately exhaustive-by-refusal: a type with no arm
/// here becomes a named error rather than a `"<unsupported>"` placeholder,
/// because a placeholder in a numbers table reads as data.
fn cell_at(
    path: &Path,
    name: &str,
    batch: &RecordBatch,
    col: usize,
    row: usize,
) -> Result<Cell, ShowError> {
    use arrow::datatypes::DataType as D;

    let array = batch.column(col);
    if array.is_null(row) {
        return Ok(Cell::Null);
    }
    let cell = match array.data_type() {
        D::Boolean => Cell::Bool(array.as_boolean().value(row)),
        D::Int8 => Cell::Int(array.as_primitive::<Int8Type>().value(row) as i64),
        D::Int16 => Cell::Int(array.as_primitive::<Int16Type>().value(row) as i64),
        D::Int32 => Cell::Int(array.as_primitive::<Int32Type>().value(row) as i64),
        D::Int64 => Cell::Int(array.as_primitive::<Int64Type>().value(row)),
        D::UInt8 => Cell::UInt(array.as_primitive::<UInt8Type>().value(row) as u64),
        D::UInt16 => Cell::UInt(array.as_primitive::<UInt16Type>().value(row) as u64),
        D::UInt32 => Cell::UInt(array.as_primitive::<UInt32Type>().value(row) as u64),
        D::UInt64 => Cell::UInt(array.as_primitive::<UInt64Type>().value(row)),
        D::Float16 => Cell::Float(array.as_primitive::<Float16Type>().value(row).to_f64()),
        D::Float32 => Cell::Float(array.as_primitive::<Float32Type>().value(row) as f64),
        D::Float64 => Cell::Float(array.as_primitive::<Float64Type>().value(row)),
        D::Utf8 => Cell::Text(array.as_string::<i32>().value(row).to_string()),
        D::LargeUtf8 => Cell::Text(array.as_string::<i64>().value(row).to_string()),
        D::Utf8View => Cell::Text(array.as_string_view().value(row).to_string()),
        other => {
            return Err(ShowError::UnsupportedColumnType {
                path: path.to_path_buf(),
                column: name.to_string(),
                data_type: format!("{other:?}"),
            });
        }
    };
    Ok(cell)
}
