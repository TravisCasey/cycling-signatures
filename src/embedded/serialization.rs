// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Wire format for [`EmbeddedTrajectory`]: an envelope recording the metric an
//! embedding was built with, and the fingerprints of the trajectory and cover
//! files it was saved alongside.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::EmbeddedTrajectory;
use crate::{
    cover::CubicalCover,
    error::{Error, Result},
    metric::Metric,
    serialization::{load_from_path, save_to_path},
    trajectory::Trajectory,
};

/// An embedding's metric, together with the fingerprints of the trajectory
/// and cover files it was saved alongside.
///
/// Establishes that the three files belong together: loading pairs a
/// trajectory and cover file with an envelope only when their fingerprints
/// match the ones the envelope recorded.
#[derive(Serialize, Deserialize)]
struct EmbeddedTrajectoryEnvelope {
    metric: Metric,
    trajectory_fingerprint: u64,
    cover_fingerprint: u64,
}

impl EmbeddedTrajectory {
    /// Writes the trajectory and the cover to `trajectory_path` and
    /// `cover_path`, then writes an envelope to `embedded_path` recording this
    /// embedding's metric and both files' fingerprints.
    ///
    /// [`load`](Self::load) reads the metric back from the envelope and
    /// verifies the trajectory and cover it loads against the recorded
    /// fingerprints, so all three files must be kept together.
    ///
    /// # Errors
    ///
    /// - [`Error::Io`] on file or serialization failure for any of the three
    ///   files.
    pub fn save<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        &self,
        embedded_path: P,
        trajectory_path: Q,
        cover_path: R,
    ) -> Result<()> {
        self.trajectory.save(trajectory_path)?;
        self.cover.save(cover_path)?;
        let envelope = EmbeddedTrajectoryEnvelope {
            metric: self.metric,
            trajectory_fingerprint: self.trajectory.fingerprint(),
            cover_fingerprint: self.cover.fingerprint(),
        };
        save_to_path(embedded_path, &envelope)
    }

    /// Reads a trajectory, a cover, and an envelope written by
    /// [`save`](Self::save), verifies both fingerprints, and reassembles an
    /// [`EmbeddedTrajectory`].
    ///
    /// The envelope establishes that `trajectory_path` and `cover_path` are
    /// the pair `embedded_path` was saved with; it does not establish that
    /// the loaded cover carries the same cohomology generator basis as any
    /// other cover built from the same cube set, since that basis is not part
    /// of a cover's fingerprint.
    ///
    /// # Errors
    ///
    /// - [`Error::EmbeddedTrajectoryFingerprintMismatch`] if the loaded
    ///   trajectory's fingerprint does not match the one the envelope recorded.
    /// - [`Error::EmbeddedCoverFingerprintMismatch`] if the loaded cover's
    ///   fingerprint does not match the one the envelope recorded.
    /// - [`Error::EmbeddedDimensionMismatch`] if the loaded trajectory and
    ///   cover disagree on spatial dimension.
    /// - [`Error::EmbeddedCubeNotInCover`] if a trajectory point maps to a cube
    ///   absent from the loaded cover.
    /// - [`Error::ConsecutiveCubesNonAdjacent`] if consecutive points of the
    ///   loaded trajectory land in cubes differing by more than 1 in some axis.
    /// - [`Error::FormatVersionMismatch`] if any of the three files' format
    ///   version differs.
    /// - [`Error::Io`] if any of the three files could not be opened.
    /// - [`Error::Deserialize`] if any of the three files' contents could not
    ///   be read and decoded.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        embedded_path: P,
        trajectory_path: Q,
        cover_path: R,
    ) -> Result<Self> {
        let envelope: EmbeddedTrajectoryEnvelope = load_from_path(embedded_path)?;

        let trajectory = Trajectory::load(trajectory_path)?;
        let trajectory_fingerprint = trajectory.fingerprint();
        if trajectory_fingerprint != envelope.trajectory_fingerprint {
            return Err(Error::EmbeddedTrajectoryFingerprintMismatch {
                expected: envelope.trajectory_fingerprint,
                found: trajectory_fingerprint,
            });
        }

        let cover = CubicalCover::load(cover_path)?;
        let cover_fingerprint = cover.fingerprint();
        if cover_fingerprint != envelope.cover_fingerprint {
            return Err(Error::EmbeddedCoverFingerprintMismatch {
                expected: envelope.cover_fingerprint,
                found: cover_fingerprint,
            });
        }

        Self::new(trajectory, cover, envelope.metric)
    }
}
