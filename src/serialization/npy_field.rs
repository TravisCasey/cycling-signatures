// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Serde adapter encoding an [`Array2`] as a `NumPy` `.npy` payload.
//!
//! The array is written as `.npy` bytes and carried as a single byte buffer, so
//! a decoder in another language recovers a typed array with one `numpy.load`.

use std::fmt;

use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, ReadableElement, WritableElement, WriteNpyExt};
use serde::{
    Deserializer, Serializer,
    de::{Error as DeserializeError, Visitor},
    ser::Error as SerializeError,
};

/// Serializes `array` as a `.npy` byte payload.
///
/// # Errors
///
/// Propagates the serializer's error, or a custom error if `.npy` encoding
/// fails.
pub(crate) fn serialize<S, A>(array: &Array2<A>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    A: WritableElement,
{
    let mut buffer: Vec<u8> = Vec::new();
    array.write_npy(&mut buffer).map_err(S::Error::custom)?;
    serializer.serialize_bytes(&buffer)
}

/// Deserializes an [`Array2`] from a `.npy` byte payload.
///
/// # Errors
///
/// Propagates the deserializer's error, or a custom error if the payload is
/// not a readable `.npy` array of the expected element type.
pub(crate) fn deserialize<'de, D, A>(deserializer: D) -> Result<Array2<A>, D::Error>
where
    D: Deserializer<'de>,
    A: ReadableElement,
{
    struct ByteBufferVisitor;

    impl Visitor<'_> for ByteBufferVisitor {
        type Value = Vec<u8>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a byte buffer holding a NumPy .npy payload")
        }

        fn visit_bytes<E: DeserializeError>(self, value: &[u8]) -> Result<Vec<u8>, E> {
            Ok(value.to_vec())
        }

        fn visit_byte_buf<E: DeserializeError>(self, value: Vec<u8>) -> Result<Vec<u8>, E> {
            Ok(value)
        }
    }

    let bytes = deserializer.deserialize_bytes(ByteBufferVisitor)?;
    Array2::<A>::read_npy(&bytes[..]).map_err(D::Error::custom)
}
