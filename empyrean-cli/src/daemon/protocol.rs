//! JSON-over-Unix-socket protocol for the empyrean daemon.
//!
//! Each message is a single line of JSON (newline-delimited).
//! The daemon reads one [`Request`], executes it, and writes one [`Response`].

use serde::{Deserialize, Serialize};

/// A request from the CLI client to the daemon.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Request {
    /// Propagate orbits to a target epoch.
    Propagate {
        object_ids: Option<Vec<String>>,
        input_path: Option<String>,
        epoch: f64,
        force_model: String,
        uncertainty_method: String,
        out_dir: String,
        format: String,
    },
    /// Generate predicted ephemeris.
    Ephemeris {
        object_ids: Option<Vec<String>>,
        input_path: Option<String>,
        observers: Vec<String>,
        epoch: f64,
        force_model: String,
        out_dir: String,
        format: String,
    },
    /// Determine orbits from ADES observations.
    Determine {
        ades_path: String,
        force_model: String,
        max_iterations: u32,
        out_dir: String,
        format: String,
    },
    /// Check if the daemon is alive.
    Ping,
    /// Shut down the daemon.
    Shutdown,
}

/// A response from the daemon to the CLI client.
#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    /// Whether the request succeeded.
    pub success: bool,
    /// Human-readable summary (printed to stderr by the client).
    pub message: String,
    /// Error detail (only when `success == false`).
    pub error: Option<String>,
}

impl Response {
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            error: None,
        }
    }

    pub fn err(message: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            error: Some(error.into()),
        }
    }
}

/// Default socket path: `~/.empyrean/empyrean.sock`.
pub fn default_socket_path() -> std::path::PathBuf {
    dirs_socket_path().unwrap_or_else(|| std::path::PathBuf::from("/tmp/empyrean.sock"))
}

fn dirs_socket_path() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(
        std::path::PathBuf::from(home)
            .join(".empyrean")
            .join("empyrean.sock"),
    )
}

/// PID file path: `~/.empyrean/empyrean.pid`.
pub fn default_pid_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(home)
        .join(".empyrean")
        .join("empyrean.pid")
}
