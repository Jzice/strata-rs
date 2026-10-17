//! The GGUF reader: memory-mapped, bounds-checked, multi-shard aware.
//!
//! Layout (v3, little-endian):
//! ```text
//! magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
//! kv_count * (key string, value type u32, value)
//! tensor_count * (name string, n_dims u32, dims u64..., type u32, offset u64)
//! [padding to `alignment`]  tensor data (offsets relative to data_start)
//! ```
//! Offsets in the tensor directory are relative to the start of the data
//! section, which begins at the first `alignment`-aligned position after the
//! directory (default alignment 32, overridable with `general.alignment`).

use crate::error::{Error, Result};
use crate::metadata::{MetaType, MetaValue, MAX_ARRAY_ITEMS_KEPT};
use crate::quant::QuantType;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
const GGUF_VERSION: u32 = 3;
const MAX_ARRAY_LEN: u64 = 1 << 24; // plausibility guard on array counts

/// A bounds-checked cursor over the mapped header region.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.pos + n > self.data.len() {
            return Err(Error::Truncated {
                expected: n,
                found: self.data.len().saturating_sub(self.pos),
            });
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn u64(&mut self) -> Result<u64> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    fn f32(&mut self) -> Result<f32> {
        self.need(4)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn f64(&mut self) -> Result<f64> {
        self.need(8)?;
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| Error::BadKeyLength { len: n })?;
        if n > 65535 {
            return Err(Error::BadKeyLength { len: n as u64 });
        }
        self.need(n)?;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + n])
            .map_err(|_| Error::StringNotUtf8)?
            .to_string();
        self.pos += n;
        Ok(s)
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn value(&mut self, ty: MetaType, depth: usize) -> Result<MetaValue> {
        if depth > 2 {
            // The reference reader refuses deeper nesting.
            return Err(Error::BadValueType { value_type: 9 });
        }
        let mut v = MetaValue {
            ty,
            int: 0,
            uint: 0,
            float: 0.0,
            string: String::new(),
            elem: MetaType::U32,
            count: 0,
            items: Vec::new(),
        };
        match ty {
            MetaType::U8 => v.int = i64::from(self.u8()?),
            MetaType::I8 => v.int = i64::from(self.i8()?),
            MetaType::U16 => v.int = i64::from(self.u16()?),
            MetaType::I16 => v.int = i64::from(self.i16()?),
            MetaType::U32 => v.int = i64::from(self.u32()?),
            MetaType::I32 => v.int = i64::from(self.i32()?),
            MetaType::F32 => v.float = f64::from(self.f32()?),
            MetaType::Bool => v.int = i64::from(self.u8()? != 0),
            MetaType::String => v.string = self.string()?,
            MetaType::U64 => v.uint = self.u64()?,
            MetaType::I64 => v.int = self.i64()?,
            MetaType::F64 => v.float = self.f64()?,
            MetaType::Array => {
                let elem = MetaType::from_id(self.u32()?)?;
                let count = self.u64()?;
                if count > MAX_ARRAY_LEN {
                    return Err(Error::BadValueType { value_type: 9 });
                }
                v.elem = elem;
                v.count = count;
                let keep = usize::try_from(count)
                    .unwrap_or(MAX_ARRAY_ITEMS_KEPT)
                    .min(MAX_ARRAY_ITEMS_KEPT.max(1));
                // Arrays are kept in full when they are small enough to matter
                // (the tokenizer's vocab/merges are the long ones); the count
                // always records the true length.
                let keep = if count <= 1 << 20 {
                    usize::try_from(count).unwrap_or(keep)
                } else {
                    MAX_ARRAY_ITEMS_KEPT
                };
                for _ in 0..keep {
                    v.items.push(self.value(elem, depth + 1)?);
                }
            }
        }
        Ok(v)
    }
}

/// One tensor's directory entry.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: dim 0 varies fastest.
    pub shape: Vec<u64>,
    pub ty: u32,
    /// Byte offset from the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn dims(&self) -> usize {
        self.shape.len()
    }
}

/// One GGUF file (one shard), memory-mapped.
pub struct GgufFile {
    path: PathBuf,
    mmap: Mmap,
    version: u32,
    alignment: usize,
    data_start: usize,
    file_size: u64,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    metadata: HashMap<String, MetaValue>,
}

#[allow(clippy::missing_fields_in_debug)] // the mapping has no useful Debug
impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The mapping itself has no useful Debug; the parsed view does.
        f.debug_struct("GgufFile")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("data_start", &self.data_start)
            .field("file_size", &self.file_size)
            .field("n_tensors", &self.tensors.len())
            .field("n_metadata", &self.metadata.len())
            .finish()
    }
}

impl GgufFile {
    /// Maps and parses one GGUF file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        // SAFETY: the mapping is read-only and outlives `mmap`; the file is
        // not mutated while mapped.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        let file_size = mmap.len() as u64;
        let mut me = Self {
            path,
            mmap,
            version: 0,
            alignment: 32,
            data_start: 0,
            file_size,
            tensors: Vec::new(),
            tensor_index: HashMap::new(),
            metadata: HashMap::new(),
        };
        me.parse()?;
        Ok(me)
    }

    fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    fn parse(&mut self) -> Result<()> {
        let mut metadata = HashMap::new();
        let mut tensors = Vec::new();
        let mut tensor_index = HashMap::new();
        let version;
        let alignment;
        let data_start;

        {
            let mut c = Cursor::new(self.bytes());
            let magic = c.u32()?;
            if magic != GGUF_MAGIC {
                return Err(Error::BadMagic { magic });
            }
            version = c.u32()?;
            if version != GGUF_VERSION {
                return Err(Error::UnsupportedVersion { version });
            }
            let n_tensors = c.u64()?;
            let n_kv = c.u64()?;

            for _ in 0..n_kv {
                let key = c.string()?;
                let ty = MetaType::from_id(c.u32()?)?;
                let value = c.value(ty, 0)?;
                metadata.insert(key, value);
            }

            let mut names = std::collections::HashSet::new();
            for _ in 0..n_tensors {
                let name = c.string()?;
                if !names.insert(name.clone()) {
                    return Err(Error::TensorNotFound { name });
                }
                let nd = c.u32()?;
                if nd == 0 || nd > 4 {
                    return Err(Error::WrongTensorDims {
                        name,
                        found: nd as usize,
                        expected: 4,
                    });
                }
                let mut shape = Vec::with_capacity(nd as usize);
                for _ in 0..nd {
                    shape.push(c.u64()?);
                }
                let ty = c.u32()?;
                let offset = c.u64()?;
                let idx = tensors.len();
                tensor_index.insert(name.clone(), idx);
                tensors.push(TensorInfo {
                    name,
                    shape,
                    ty,
                    offset,
                });
            }

            let mut align: u64 = 32;
            if let Some(a) = metadata.get("general.alignment").and_then(MetaValue::as_int) {
                if a > 0 {
                    align = a as u64;
                }
            }
            alignment = usize::try_from(align).unwrap_or(32).max(1);
            data_start = c.pos().div_ceil(alignment) * alignment;
        }

        let file_len = self.bytes().len();
        let data_start = if data_start > file_len {
            if !tensors.is_empty() {
                return Err(Error::Truncated {
                    expected: data_start,
                    found: file_len,
                });
            }
            // A metadata-only shard (a split model's first shard) may end
            // before the aligned data start; clamp it.
            file_len
        } else {
            data_start
        };

        self.version = version;
        self.alignment = alignment;
        self.data_start = data_start;
        self.metadata = metadata;
        self.tensors = tensors;
        self.tensor_index = tensor_index;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    pub fn data_start(&self) -> usize {
        self.data_start
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn metadata(&self) -> &HashMap<String, MetaValue> {
        &self.metadata
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// How many tensors of the given type the file holds.
    pub fn count_type(&self, ty: u32) -> usize {
        self.tensors.iter().filter(|t| t.ty == ty).count()
    }

    /// The tensor's payload bytes, exactly the bytes its shape and type need.
    /// An error when the type is unknown, a row is not whole blocks, or the
    /// count overflows.
    pub fn tensor_payload_bytes(t: &TensorInfo) -> Result<usize> {
        if t.shape.is_empty() {
            return Err(Error::WrongTensorDims {
                name: t.name.clone(),
                found: 0,
                expected: 4,
            });
        }
        let quant = QuantType::from_id(t.ty)?;
        let n_blocks = t.shape.iter().try_fold(1u64, |acc, d| {
            if *d == 0 || acc > u64::MAX / d {
                Err(Error::Truncated {
                    expected: t.offset as usize,
                    found: 0,
                })
            } else {
                Ok(acc * d)
            }
        })? / quant.block_size() as u64;
        usize::try_from(n_blocks * quant.type_size() as u64).map_err(|_| Error::Truncated {
            expected: t.offset as usize,
            found: 0,
        })
    }

    /// The tensor's payload bytes, bounds-checked against the file.
    pub fn tensor_data(&self, t: &TensorInfo) -> Result<&[u8]> {
        let bytes = Self::tensor_payload_bytes(t)?;
        let start = self
            .data_start
            .checked_add(t.offset as usize)
            .ok_or(Error::Truncated {
                expected: t.offset as usize,
                found: 0,
            })?;
        let end = start.checked_add(bytes).ok_or(Error::Truncated {
            expected: bytes,
            found: 0,
        })?;
        if end > self.bytes().len() {
            return Err(Error::Truncated {
                expected: bytes,
                found: self.bytes().len().saturating_sub(start),
            });
        }
        Ok(&self.bytes()[start..end])
    }
}

/// A whole model: every shard of a split GGUF, opened together, with one
/// name index across them.
pub struct GgufModel {
    shards: Vec<GgufFile>,
    index: HashMap<String, (usize, usize)>, // name -> (shard, tensor)
}

impl std::fmt::Debug for GgufModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufModel")
            .field("n_shards", &self.shards.len())
            .field("n_tensors", &self.index.len())
            .finish()
    }
}

impl GgufModel {
    /// Opens the shards of the model that `any_shard` belongs to.
    pub fn open(any_shard: impl AsRef<Path>) -> Result<Self> {
        let paths = split_paths(any_shard.as_ref())?;
        let mut shards = Vec::with_capacity(paths.len());
        for p in &paths {
            shards.push(GgufFile::open(p)?);
        }
        let mut index = HashMap::new();
        for (si, shard) in shards.iter().enumerate() {
            for (ti, t) in shard.tensors().iter().enumerate() {
                if index.insert(t.name.clone(), (si, ti)).is_some() {
                    return Err(Error::TensorNotFound { name: t.name.clone() });
                }
            }
        }
        Ok(Self { shards, index })
    }

    pub fn shards(&self) -> &[GgufFile] {
        &self.shards
    }

    /// The metadata shard (the first one).
    pub fn meta(&self) -> &GgufFile {
        &self.shards[0]
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.shards[0].get(key)
    }

    /// The tensor named `name`, with its shard index.
    pub fn find(&self, name: &str) -> Option<(&TensorInfo, usize)> {
        self.index
            .get(name)
            .map(|&(si, ti)| (&self.shards[si].tensors()[ti], si))
    }

    /// The tensor's payload bytes from whichever shard holds it.
    pub fn tensor_data(&self, name: &str) -> Option<Result<&[u8]>> {
        let (t, si) = self.find(name)?;
        Some(self.shards[si].tensor_data(t))
    }

    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }

    /// Every tensor across every shard.
    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }

    /// The mapped bytes of every shard: what the model costs to keep open.
    pub fn mapped_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.bytes().len() as u64).sum()
    }
}

/// The shards of a split model: `<name>-NNNNN-of-MMMMM.gguf` ... in split
/// order (the metadata shard, 00001, first). A path without that suffix is a
/// single-shard model. A missing shard is an error that names it.
pub fn split_paths(any: &Path) -> Result<Vec<PathBuf>> {
    let s = any.to_string_lossy().to_string();
    let Some(at) = s.rfind("-of-") else {
        return Ok(vec![any.to_path_buf()]);
    };
    let bytes = s.as_bytes();
    // "...-NNNNN-of-MMMMM.gguf": five digits before the tag, five after it,
    // then the extension.
    let ok_shape =
        at >= 6 && bytes[at - 6] == b'-' && s.len() == at + 4 + 5 + 5 && &s[at + 9..at + 14] == ".gguf";
    if !ok_shape {
        return Ok(vec![any.to_path_buf()]);
    }
    let mut no = 0usize;
    let mut n = 0usize;
    for i in 0..5 {
        let a = bytes[at - 5 + i];
        let b = bytes[at + 4 + i];
        if !a.is_ascii_digit() || !b.is_ascii_digit() {
            return Ok(vec![any.to_path_buf()]);
        }
        no = no * 10 + usize::from(a - b'0');
        n = n * 10 + usize::from(b - b'0');
    }
    if n < 1 || no < 1 || no > n {
        return Ok(vec![any.to_path_buf()]);
    }
    let mut out = Vec::with_capacity(n);
    for i in 1..=n {
        let mut p = s.clone();
        let num = format!("{i:05}");
        p.replace_range(at - 5..at, &num);
        let path = PathBuf::from(p);
        if !path.exists() {
            return Err(Error::Io {
                path: path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("missing model shard {i} of {n} (is the download complete?)"),
                ),
            });
        }
        out.push(path);
    }
    Ok(out)
}
