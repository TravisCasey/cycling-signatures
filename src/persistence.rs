// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Wire-format glue for persisted artifacts.
//!
//! Defines the on-disk format version, the version envelope every file
//! carries, and the read/write helpers shared by the save/load methods on the
//! persistable types. The container is `MessagePack` with structs encoded as
//! field-name maps; large arrays ride inside as `.npy` payloads via
//! [`npy_field`].

use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use serde::{Serialize, de::DeserializeOwned};

use crate::error::{Error, Result};

pub(crate) mod npy_field;

/// The on-disk format version. Every persisted file carries it; loaders refuse
/// a value other than this one.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// The envelope wrapping every persisted payload with its format version.
#[derive(Serialize, serde::Deserialize)]
pub(crate) struct Versioned<T> {
    pub(crate) format_version: u32,
    pub(crate) payload: T,
}

/// Writes `payload` to `path` wrapped in the current-version envelope, as
/// `MessagePack` with struct-map encoding.
///
/// # Errors
///
/// [`Error::Storage`] on file or serialization failure.
pub(crate) fn save_to_path<T, P>(path: P, payload: &T) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let versioned = Versioned {
        format_version: FORMAT_VERSION,
        payload,
    };
    rmp_serde::encode::write_named(&mut writer, &versioned)?;
    writer.flush()?;
    Ok(())
}

/// Reads a payload written by [`save_to_path`], verifying the format version.
///
/// # Errors
///
/// - [`Error::FormatVersionMismatch`] if the file's version differs.
/// - [`Error::Storage`] on file or deserialization failure.
pub(crate) fn load_from_path<T, P>(path: P) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let versioned: Versioned<T> = rmp_serde::decode::from_read(reader)?;
    if versioned.format_version != FORMAT_VERSION {
        return Err(Error::FormatVersionMismatch {
            expected: FORMAT_VERSION,
            found: versioned.format_version,
        });
    }
    Ok(versioned.payload)
}
