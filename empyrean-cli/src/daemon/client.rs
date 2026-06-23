//! Daemon client — tries to connect to a running daemon.
//!
//! If the daemon is available, sends the request and returns the response.
//! If not, returns `None` so the caller can fall back to in-process execution.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use crate::daemon::protocol::{self, Request, Response};

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Try to send a request to the daemon. Returns `None` if no daemon is running.
///
/// If the socket exists but the daemon is still loading, shows a spinner
/// and retries the connection until the daemon is ready (up to 120s).
pub fn try_request(request: &Request) -> Option<Response> {
    let socket_path = protocol::default_socket_path();
    if !socket_path.exists() {
        return None;
    }

    // Try to connect. The daemon binds the socket before loading context,
    // so the connect() succeeds immediately but the response may take a
    // while as the daemon finishes loading.
    let stream = match UnixStream::connect(&socket_path) {
        Ok(s) => s,
        Err(_) => {
            // Socket exists but connect failed — daemon may be starting up.
            // Retry with a spinner.
            let start = Instant::now();
            let timeout = Duration::from_secs(120);
            let mut spin_idx = 0;
            loop {
                eprint!(
                    "\r{} Waiting for daemon...",
                    SPINNER[spin_idx % SPINNER.len()]
                );
                spin_idx += 1;
                std::thread::sleep(Duration::from_millis(100));

                if let Ok(s) = UnixStream::connect(&socket_path) {
                    eprint!("\r                        \r");
                    break s;
                }
                if start.elapsed() > timeout {
                    eprint!("\r                        \r");
                    return None;
                }
            }
        }
    };

    // Connected — daemon is listening. Send request and wait for response.
    // The daemon may still be loading context, so the read timeout is generous.
    stream
        .set_read_timeout(Some(Duration::from_secs(600)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .ok()?;

    let json = serde_json::to_string(request).ok()?;
    let mut writer = &stream;
    writeln!(writer, "{json}").ok()?;

    // Show spinner while waiting for response (context may still be loading).
    stream.set_nonblocking(true).ok()?;
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    let mut spin_idx = 0;

    loop {
        match reader.read_line(&mut line) {
            Ok(0) => return None, // EOF
            Ok(_) => {
                eprint!("\r                        \r");
                break;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                eprint!(
                    "\r{} Waiting for daemon...",
                    SPINNER[spin_idx % SPINNER.len()]
                );
                spin_idx += 1;
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => return None,
        }
    }

    serde_json::from_str(line.trim()).ok()
}

/// Check if a daemon is running by sending a Ping.
#[allow(dead_code)]
pub fn is_daemon_running() -> bool {
    try_request(&Request::Ping)
        .map(|r| r.success)
        .unwrap_or(false)
}
