//! GGUF writer. See docs/convert-design.md.
//! Writes a GGUF v3 file from header, metadata, tensor table, and data.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::gguf::{GgufError, GgufTensorInfo, GgufType, GgufValue, GGUF_MAGIC, GGUF_VERSION_3};

fn write_value<W: Write>(w: &mut W, v: &GgufValue) -> std::io::Result<()> {
    match v {
        GgufValue::Uint8(x) => { w.write_all(&0u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GgufValue::Int8(x) => { w.write_all(&1u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GgufValue::Uint16(x) => { w.write_all(&2u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GgufValue::Int16(x) => { w.write_all(&3u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GgufValue::Uint32(x) => { w.write_all(&4u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GgufValue::Int32(x) => { w.write_all(&5u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GgufValue::Float32(x) => { w.write_all(&6u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GgufValue::Bool(x) => { w.write_all(&7u32.to_le_bytes())?; w.write_all(&[if *x {1u8} else {0u8}])?; }
        GgufValue::String(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            let bytes = s.as_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)?;
        }
        GgufValue::Array { element_type, values } => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&(*element_type as u32).to_le_bytes())?;
            w.write_all(&(values.len() as u64).to_le_bytes())?;
            for item in values {
                write_payload(w, item)?;
            }
        }
    }
    Ok(())
}

fn write_payload<W: Write>(w: &mut W, v: &GgufValue) -> std::io::Result<()> {
    match v {
        GgufValue::Uint8(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::Int8(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::Uint16(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::Int16(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::Uint32(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::Int32(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::Float32(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::Bool(x) => w.write_all(&[if *x {1u8} else {0u8}]),
        GgufValue::String(s) => {
            let bytes = s.as_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)
        }
        GgufValue::Array { .. } => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "nested arrays are not supported in GGUF",
        )),
    }
}

fn dtype_to_u32(t: GgufType) -> u32 {
    match t {
        GgufType::F32 => 0,
        GgufType::F16 => 1,
        GgufType::Q4_0 => 2,
        GgufType::Q4_1 => 3,
        GgufType::Q5_0 => 6,
        GgufType::Q5_1 => 7,
        GgufType::Q8_0 => 8,
        GgufType::Q8_1 => 9,
        GgufType::Q2_K => 10,
        GgufType::Q3_K => 11,
        GgufType::Q4_K => 12,
        GgufType::Q5_K => 13,
        GgufType::Q6_K => 14,
        GgufType::Q8_K => 15,
        GgufType::Tq1_0 => 1000,
        GgufType::Unknown(v) => v,
    }
}

const ALIGNMENT: u64 = 32;

/// Write a GGUF v3 file.
pub fn write_gguf(
    path: &Path,
    metadata: &[(String, GgufValue)],
    tensors: &[(GgufTensorInfo, Vec<u8>)],
) -> Result<(), GgufError> {
    let f = File::create(path).map_err(GgufError::Io)?;
    let mut w = BufWriter::new(f);

    // Header.
    w.write_all(&GGUF_MAGIC.to_le_bytes()).map_err(GgufError::Io)?;
    w.write_all(&GGUF_VERSION_3.to_le_bytes()).map_err(GgufError::Io)?;
    w.write_all(&(tensors.len() as u64).to_le_bytes()).map_err(GgufError::Io)?;
    w.write_all(&(metadata.len() as u64).to_le_bytes()).map_err(GgufError::Io)?;

    // Metadata.
    for (k, v) in metadata {
        let kb = k.as_bytes();
        w.write_all(&(kb.len() as u64).to_le_bytes()).map_err(GgufError::Io)?;
        w.write_all(kb).map_err(GgufError::Io)?;
        write_value(&mut w, v).map_err(GgufError::Io)?;
    }

    // Tensor table: name, ndims, dims, dtype, offset.
    // We compute offsets during a first pass with the current writer
    // position, then align the data section.
    // To avoid buffering all data, write the table with placeholder
    // offsets, then fix up.
    let table_start_pos;
    {
        // We need to know the current position to compute offsets.
        // BufWriter does not expose it, so flush first.
        w.flush().map_err(GgufError::Io)?;
        table_start_pos = w.get_ref().stream_position().map_err(GgufError::Io)?;
    }
    // Compute table byte size.
    let mut table_size: u64 = 0;
    for (t, _) in tensors {
        table_size += 8 + t.name.len() as u64;      // name len + name
        table_size += 4;                             // ndims
        table_size += 8 * t.shape.len() as u64;      // dims
        table_size += 4;                             // dtype
        table_size += 8;                             // offset
    }
    let data_start_unaligned = table_start_pos + table_size;
    let data_start = (data_start_unaligned + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;

    // Write table with real offsets.
    let mut cursor = data_start;
    for (t, data) in tensors {
        let nb = t.name.as_bytes();
        w.write_all(&(nb.len() as u64).to_le_bytes()).map_err(GgufError::Io)?;
        w.write_all(nb).map_err(GgufError::Io)?;
        w.write_all(&(t.shape.len() as u32).to_le_bytes()).map_err(GgufError::Io)?;
        for d in &t.shape {
            w.write_all(&d.to_le_bytes()).map_err(GgufError::Io)?;
        }
        w.write_all(&dtype_to_u32(t.dtype).to_le_bytes()).map_err(GgufError::Io)?;
        w.write_all(&(cursor - data_start).to_le_bytes()).map_err(GgufError::Io)?;
        cursor += data.len() as u64;
        // Align each tensor to ALIGNMENT.
        let rem = cursor % ALIGNMENT;
        if rem != 0 { cursor += ALIGNMENT - rem; }
    }

    // Pad to data_start.
    w.flush().map_err(GgufError::Io)?;
    let cur = w.get_ref().stream_position().map_err(GgufError::Io)?;
    for _ in cur..data_start {
        w.write_all(&[0u8]).map_err(GgufError::Io)?;
    }

    // Write tensor data with alignment padding.
    let mut written: u64 = 0;
    for (_, data) in tensors {
        w.write_all(data).map_err(GgufError::Io)?;
        written += data.len() as u64;
        let rem = written % ALIGNMENT;
        if rem != 0 {
            let pad = ALIGNMENT - rem;
            for _ in 0..pad { w.write_all(&[0u8]).map_err(GgufError::Io)?; }
            written += pad;
        }
    }

    w.flush().map_err(GgufError::Io)?;
    Ok(())
}
