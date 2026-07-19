// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Wire-format glue for saved artifacts.
//!
//! Defines the on-disk format version, the version envelope every file
//! carries, and the read/write helpers shared by the save/load methods on the
//! serializable types. The container is `MessagePack` with structs encoded as
//! field-name maps; large arrays ride inside as `.npy` payloads via
//! [`npy_field`].

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::error::{Error, Result};

pub(crate) mod npy_field;

/// The on-disk format version. Every saved file carries it; loaders refuse
/// a value other than this one.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// The envelope wrapping every saved payload with its format version.
#[derive(Serialize, Deserialize)]
pub(crate) struct Versioned<T> {
    pub(crate) format_version: u32,
    pub(crate) payload: T,
}

/// Writes `payload` to `writer` wrapped in the current-version envelope, as
/// `MessagePack` with struct-map encoding.
///
/// # Errors
///
/// [`Error::Io`] on serialization or input/output failure.
pub(crate) fn save_to_writer<T, W>(mut writer: W, payload: &T) -> Result<()>
where
    T: Serialize,
    W: Write,
{
    let versioned = Versioned {
        format_version: FORMAT_VERSION,
        payload,
    };
    rmp_serde::encode::write_named(&mut writer, &versioned)?;
    writer.flush()?;
    Ok(())
}

/// Reads a payload written by [`save_to_writer`], verifying the format version.
///
/// # Errors
///
/// - [`Error::FormatVersionMismatch`] if the payload's version differs.
/// - [`Error::Deserialize`] if the payload could not be read and decoded.
pub(crate) fn load_from_reader<T, R>(reader: R) -> Result<T>
where
    T: DeserializeOwned,
    R: Read,
{
    let versioned: Versioned<T> = rmp_serde::decode::from_read(reader)?;
    if versioned.format_version != FORMAT_VERSION {
        return Err(Error::FormatVersionMismatch {
            expected: FORMAT_VERSION,
            found: versioned.format_version,
        });
    }
    Ok(versioned.payload)
}

/// Writes `payload` to `path` in the crate's binary format.
///
/// # Errors
///
/// [`Error::Io`] on file or serialization failure.
pub(crate) fn save_to_path<T, P>(path: P, payload: &T) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let file = File::create(path)?;
    save_to_writer(BufWriter::new(file), payload)
}

/// Reads a payload written by [`save_to_path`], verifying the format version.
///
/// # Errors
///
/// - [`Error::FormatVersionMismatch`] if the payload's version differs.
/// - [`Error::Io`] if the file could not be opened.
/// - [`Error::Deserialize`] if the file contents could not be read and decoded.
pub(crate) fn load_from_path<T, P>(path: P) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    load_from_reader(BufReader::new(file))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A writer whose every operation fails.
    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("writer failure"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("writer failure"))
        }
    }

    #[test]
    fn failing_writer_reports_io_error() {
        let result = save_to_writer(FailingWriter, &7_u32);
        assert!(matches!(result, Err(Error::Io { .. })));
    }

    #[test]
    fn malformed_payload_reports_deserialize_error() {
        let malformed: &[u8] = b"not a MessagePack envelope";
        let result = load_from_reader::<u32, _>(malformed);
        assert!(matches!(result, Err(Error::Deserialize { .. })));
    }
}
