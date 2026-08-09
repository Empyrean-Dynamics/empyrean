//! The interactive pager.
//!
//! Split in two on purpose. [`Viewport`] and [`classify_key`] are pure —
//! they take a key event and a scroll position and produce the next
//! scroll position, with no terminal anywhere — so the whole paging state
//! machine is unit-tested by feeding synthetic key events. Only
//! [`run`] touches crossterm.
//!
//! ## Why backward paging re-reads the file
//!
//! Rows arrive from a stream that cannot seek to an arbitrary row: a
//! Parquet reader is positioned by row group and page, a CSV reader by
//! quote state. The honest ways to make `b` cheap are to buffer every row
//! seen so far (unbounded memory — the thing this pager exists to avoid)
//! or to index the file up front (reading all of it before the first page
//! draws). So `b` restarts the stream and skips forward to the target
//! row: **O(rows above the target)** per press, with no rows retained.
//! On a local Parquet file that is milliseconds for the first thousands
//! of rows and seconds once you are deep into millions. Forward paging
//! does not pay this — the open stream is kept and pulled from — so
//! walking forward through a huge file stays flat.

use std::io::{IsTerminal, Write};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::{cursor, execute, queue, style, terminal};

use super::error::ShowError;
use super::render::{self, Widths};
use super::table::Table;

/// What a keypress means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    NextPage,
    PrevPage,
    ColumnsLeft,
    ColumnsRight,
    /// Jump back to the first row.
    Home,
    /// Begin typing a row filter.
    BeginFilter,
    /// Drop the active filter.
    ClearFilter,
    Quit,
    /// A key with no binding — redraw nothing, wait for the next one.
    Ignore,
}

/// Map a key event to an action.
///
/// `more`-style bindings: space and enter advance, `b` goes back, the
/// arrows slide the column window, `/` filters, `q` quits. Ctrl-C is a
/// quit too, since a pager that ignores it is a trap.
///
/// Ctrl-D is deliberately **not** a quit. In `more` and `less` it scrolls,
/// so a reader who presses it expects to move down the file, not to be
/// dropped out of it — and some terminal harnesses emit one unbidden.
/// Whether a key means "Enter".
///
/// A terminal may deliver Enter as CR (0x0D), which crossterm reports as
/// [`KeyCode::Enter`], or as LF (0x0A), which it reports as Ctrl-J — the
/// two are the same key historically, and which one arrives depends on
/// the line discipline of whatever pty is in the way (`script`, `tmux`,
/// an ssh session). Accepting only the first leaves the filter prompt
/// with no way to confirm, and it then swallows every following keypress.
pub fn is_enter(key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Enter => true,
        KeyCode::Char('j') | KeyCode::Char('m') => key.modifiers.contains(KeyModifiers::CONTROL),
        _ => false,
    }
}

pub fn classify_key(key: KeyEvent) -> Action {
    // Key *releases* arrive on some terminals; acting on both edges would
    // page twice per press.
    if key.kind == KeyEventKind::Release {
        return Action::Ignore;
    }
    if is_enter(key) {
        return Action::NextPage;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c') => Action::Quit,
            _ => Action::Ignore,
        };
    }
    match key.code {
        KeyCode::Char(' ') | KeyCode::Enter | KeyCode::PageDown | KeyCode::Char('f') => {
            Action::NextPage
        }
        KeyCode::Char('b') | KeyCode::PageUp => Action::PrevPage,
        KeyCode::Left | KeyCode::Char('h') => Action::ColumnsLeft,
        KeyCode::Right | KeyCode::Char('l') => Action::ColumnsRight,
        KeyCode::Home | KeyCode::Char('g') => Action::Home,
        KeyCode::Char('/') => Action::BeginFilter,
        KeyCode::Esc => Action::ClearFilter,
        KeyCode::Char('q') => Action::Quit,
        _ => Action::Ignore,
    }
}

/// Where the pager is looking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Viewport {
    /// Index of the first displayed row, counted over *filtered* rows.
    pub top: usize,
    /// How many rows fit below the header.
    pub page_rows: usize,
    /// Index of the leftmost displayed column.
    pub col_start: usize,
    /// Total filtered rows, once the end has been reached. Until then the
    /// pager does not know it, and must not pretend to.
    pub total_rows: Option<usize>,
}

impl Viewport {
    pub fn new(page_rows: usize) -> Self {
        Self {
            top: 0,
            page_rows: page_rows.max(1),
            col_start: 0,
            total_rows: None,
        }
    }

    /// Apply an action. Returns whether the screen needs redrawing.
    ///
    /// `max_col_start` comes from the current widths and terminal width;
    /// `rows_on_page` is how many rows the last draw actually produced,
    /// which is how a short page teaches the viewport where the end is.
    pub fn apply(&mut self, action: Action, max_col_start: usize) -> bool {
        match action {
            Action::NextPage => {
                let next = self.top + self.page_rows;
                // Never scroll past the last row. When the total is not
                // yet known, advancing is always allowed — the draw that
                // follows discovers the end and clamps.
                match self.total_rows {
                    Some(total) if next >= total => false,
                    _ => {
                        self.top = next;
                        true
                    }
                }
            }
            Action::PrevPage => {
                if self.top == 0 {
                    return false;
                }
                self.top = self.top.saturating_sub(self.page_rows);
                true
            }
            Action::ColumnsLeft => {
                if self.col_start == 0 {
                    return false;
                }
                self.col_start -= 1;
                true
            }
            Action::ColumnsRight => {
                if self.col_start >= max_col_start {
                    return false;
                }
                self.col_start += 1;
                true
            }
            Action::Home => {
                if self.top == 0 && self.col_start == 0 {
                    return false;
                }
                self.top = 0;
                self.col_start = 0;
                true
            }
            Action::BeginFilter | Action::ClearFilter | Action::Quit | Action::Ignore => false,
        }
    }

    /// Record what a draw produced: a page shorter than `page_rows` means
    /// the stream ended, which fixes the total row count.
    ///
    /// Returns `true` when the viewport had to be pulled back because it
    /// had scrolled past the end (possible when `NextPage` was applied
    /// before the total was known).
    pub fn observe_page(&mut self, rows_drawn: usize) -> bool {
        if rows_drawn == self.page_rows {
            return false;
        }
        let total = self.top + rows_drawn;
        self.total_rows = Some(total);
        if rows_drawn == 0 && self.top > 0 {
            // Scrolled off the end entirely: step back to the last page
            // that has rows on it.
            self.top = total.saturating_sub(1) / self.page_rows * self.page_rows;
            return true;
        }
        false
    }

    /// A `123-145 of 200` / `123-145` position string for the status bar.
    pub fn position(&self, rows_drawn: usize) -> String {
        if rows_drawn == 0 {
            return match self.total_rows {
                Some(0) => "0 rows".to_string(),
                _ => "no matching rows".to_string(),
            };
        }
        let first = self.top + 1;
        let last = self.top + rows_drawn;
        match self.total_rows {
            Some(total) => format!("rows {first}-{last} of {total}"),
            None => format!("rows {first}-{last}"),
        }
    }
}

/// Lines the pager spends on chrome rather than data: the status bar
/// always, plus the header and the rule under it unless `--no-header`
/// suppressed them — in which case those two lines go back to the table.
fn chrome_lines(show_header: bool) -> u16 {
    if show_header { 3 } else { 1 }
}

/// The size assumed when the terminal will not say.
const FALLBACK_SIZE: (u16, u16) = (80, 24);

/// Terminal size, with zeroes treated as "unknown".
///
/// Some pty setups report 0×0 rather than failing the ioctl. Taken at
/// face value that collapses the table to a single truncated column, so
/// a zero in either axis falls back to a conventional terminal.
fn terminal_size() -> (u16, u16) {
    let (cols, rows) = terminal::size().unwrap_or(FALLBACK_SIZE);
    (
        if cols == 0 { FALLBACK_SIZE.0 } else { cols },
        if rows == 0 { FALLBACK_SIZE.1 } else { rows },
    )
}

/// Run the interactive pager over `table`.
///
/// `show_header` is `--no-header` inverted. It is honoured here and not
/// quietly ignored: a flag that does nothing in half the modes it is
/// accepted in is a flag that lies.
pub fn run(mut table: Table, show_header: bool) -> Result<(), ShowError> {
    let (mut cols, mut rows) = terminal_size();
    let mut widths = Widths::from_header(table.header());
    let mut viewport = Viewport::new(page_rows(rows, show_header));

    terminal::enable_raw_mode()
        .map_err(|e| ShowError::io(table.path(), "enter raw mode for", e))?;
    let outcome = interact(
        &mut table,
        &mut viewport,
        &mut widths,
        &mut cols,
        &mut rows,
        show_header,
    );
    // Always leave the terminal as it was found, even on the error path.
    let _ = terminal::disable_raw_mode();
    let mut out = std::io::stdout();
    let _ = execute!(out, cursor::Show);
    let _ = writeln!(out);
    outcome
}

fn page_rows(terminal_rows: u16, show_header: bool) -> usize {
    (terminal_rows.saturating_sub(chrome_lines(show_header))).max(1) as usize
}

fn interact(
    table: &mut Table,
    viewport: &mut Viewport,
    widths: &mut Widths,
    cols: &mut u16,
    rows: &mut u16,
    show_header: bool,
) -> Result<(), ShowError> {
    // The stream currently open, and the row index just past its last
    // delivered row. Forward paging reuses it; anything else restarts.
    let mut cursor_row = 0_usize;
    let mut stream = table.restart()?;
    let mut status_note: Option<String> = None;

    loop {
        // Position the stream at `viewport.top`, restarting if the target
        // is behind where the open stream already is.
        if cursor_row > viewport.top {
            stream = table.restart()?;
            cursor_row = 0;
        }
        if cursor_row < viewport.top {
            cursor_row += stream.skip(viewport.top - cursor_row)?;
        }
        let page = stream.take(viewport.page_rows)?;
        cursor_row += page.len();

        if viewport.observe_page(page.len()) {
            // The page was empty because we had scrolled off the end;
            // `observe_page` pulled `top` back, so redraw from there.
            continue;
        }
        for row in &page {
            widths.observe(row);
        }

        let max_start = render::max_column_start(widths.as_slice(), *cols as usize);
        viewport.col_start = viewport.col_start.min(max_start);
        draw(
            table,
            viewport,
            widths,
            &page,
            *cols,
            max_start,
            status_note.take(),
            show_header,
        )?;

        match read_action(table, viewport, cols, rows, max_start, show_header)? {
            Step::Quit => return Ok(()),
            Step::Redraw => {}
            Step::Reopen(note) => {
                stream = table.restart()?;
                cursor_row = 0;
                *widths = Widths::from_header(table.header());
                status_note = note;
            }
        }
    }
}

enum Step {
    Quit,
    Redraw,
    Reopen(Option<String>),
}

fn read_action(
    table: &mut Table,
    viewport: &mut Viewport,
    cols: &mut u16,
    rows: &mut u16,
    max_col_start: usize,
    show_header: bool,
) -> Result<Step, ShowError> {
    loop {
        let ev = event::read().map_err(|e| ShowError::io(table.path(), "read a key for", e))?;
        match ev {
            Event::Resize(w, h) => {
                let (w, h) = (
                    if w == 0 { FALLBACK_SIZE.0 } else { w },
                    if h == 0 { FALLBACK_SIZE.1 } else { h },
                );
                *cols = w;
                *rows = h;
                viewport.page_rows = page_rows(h, show_header);
                // The row the user is looking at is the anchor; keep it.
                return Ok(Step::Redraw);
            }
            Event::Key(key) => match classify_key(key) {
                Action::Quit => return Ok(Step::Quit),
                Action::Ignore => continue,
                Action::BeginFilter => {
                    let Some(entered) = prompt_filter(table, *rows)? else {
                        return Ok(Step::Redraw);
                    };
                    let mut view = table.view().clone();
                    view.filter = (!entered.is_empty()).then_some(entered.clone());
                    *table = Table::open(table.path(), view)?;
                    *viewport = Viewport::new(viewport.page_rows);
                    let note = if entered.is_empty() {
                        None
                    } else {
                        Some(format!("filter: {entered}"))
                    };
                    return Ok(Step::Reopen(note));
                }
                Action::ClearFilter => {
                    if table.view().filter.is_none() {
                        continue;
                    }
                    let mut view = table.view().clone();
                    view.filter = None;
                    *table = Table::open(table.path(), view)?;
                    *viewport = Viewport::new(viewport.page_rows);
                    return Ok(Step::Reopen(Some("filter cleared".to_string())));
                }
                action => {
                    if viewport.apply(action, max_col_start) {
                        return Ok(Step::Redraw);
                    }
                    // A no-op key — already at the top, already showing
                    // the last column — must not repaint the screen, or
                    // holding an arrow down flickers forever.
                    continue;
                }
            },
            _ => continue,
        }
    }
}

/// Read a filter string on the status line, with raw mode still on.
///
/// Returns `None` if the user pressed escape, `Some("")` if they
/// confirmed an empty string (which clears the filter).
fn prompt_filter(table: &Table, rows: u16) -> Result<Option<String>, ShowError> {
    let mut out = std::io::stdout();
    let mut buf = String::new();
    let io_err = |e| ShowError::io(table.path(), "prompt on the terminal for", e);
    loop {
        queue!(
            out,
            cursor::MoveTo(0, rows.saturating_sub(1)),
            terminal::Clear(terminal::ClearType::CurrentLine),
            style::Print(format!("/{buf}")),
            cursor::Show,
        )
        .map_err(io_err)?;
        out.flush().map_err(io_err)?;

        let Event::Key(key) = event::read().map_err(io_err)? else {
            continue;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        if is_enter(key) {
            return Ok(Some(buf));
        }
        match key.code {
            KeyCode::Esc => return Ok(None),
            KeyCode::Backspace => {
                buf.pop();
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Ok(None);
            }
            KeyCode::Char(c) => buf.push(c),
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw(
    table: &Table,
    viewport: &Viewport,
    widths: &Widths,
    page: &[Vec<String>],
    cols: u16,
    max_start: usize,
    note: Option<String>,
    show_header: bool,
) -> Result<(), ShowError> {
    let mut out = std::io::stdout();
    let io_err = |e| ShowError::io(table.path(), "draw to the terminal for", e);
    let width = cols as usize;
    let range = render::visible_columns(widths.as_slice(), viewport.col_start, width);

    queue!(
        out,
        cursor::Hide,
        terminal::Clear(terminal::ClearType::All),
        cursor::MoveTo(0, 0)
    )
    .map_err(io_err)?;

    if show_header {
        let header = render::draw_row(table.header(), widths.as_slice(), range);
        queue!(out, style::Print(&header), cursor::MoveToNextLine(1)).map_err(io_err)?;
        queue!(
            out,
            style::Print("─".repeat(header.chars().count().min(width))),
            cursor::MoveToNextLine(1)
        )
        .map_err(io_err)?;
    }

    for row in page {
        queue!(
            out,
            style::Print(render::draw_row(row, widths.as_slice(), range)),
            cursor::MoveToNextLine(1)
        )
        .map_err(io_err)?;
    }

    let status = status_line(table, viewport, page.len(), range, max_start, note);
    queue!(
        out,
        cursor::MoveTo(0, status_row(viewport.page_rows, show_header)),
        terminal::Clear(terminal::ClearType::CurrentLine),
        style::Print(render::fit(&status, width).trim_end().to_string()),
    )
    .map_err(io_err)?;
    out.flush().map_err(io_err)?;
    Ok(())
}

fn status_row(page_rows: usize, show_header: bool) -> u16 {
    (page_rows as u16).saturating_add(chrome_lines(show_header) - 1)
}

fn status_line(
    table: &Table,
    viewport: &Viewport,
    rows_drawn: usize,
    range: (usize, usize),
    max_start: usize,
    note: Option<String>,
) -> String {
    let (start, end) = range;
    let ncols = table.header().len();
    let mut parts = vec![viewport.position(rows_drawn)];
    if end - start < ncols {
        parts.push(format!("cols {}-{} of {ncols}", start + 1, end));
    }
    if let Some(filter) = table.view().filter.as_deref() {
        parts.push(format!("/{filter}"));
    }
    if let Some(note) = note {
        parts.push(note);
    }
    let keys = if max_start > 0 {
        "space/b page  ←/→ cols  / filter  q quit"
    } else {
        "space/b page  / filter  q quit"
    };
    format!("{}  —  {keys}", parts.join("  ·  "))
}

/// Whether an interactive pager is possible at all.
///
/// Needs both ends: a terminal to draw on and a terminal to read keys
/// from. Missing either, `show` streams the table out plainly instead.
pub fn is_interactive() -> bool {
    std::io::stdout().is_terminal() && std::io::stdin().is_terminal()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn more_style_keys_are_bound() {
        assert_eq!(classify_key(key(KeyCode::Char(' '))), Action::NextPage);
        assert_eq!(classify_key(key(KeyCode::Enter)), Action::NextPage);
        assert_eq!(classify_key(key(KeyCode::PageDown)), Action::NextPage);
        assert_eq!(classify_key(key(KeyCode::Char('b'))), Action::PrevPage);
        assert_eq!(classify_key(key(KeyCode::PageUp)), Action::PrevPage);
        assert_eq!(classify_key(key(KeyCode::Left)), Action::ColumnsLeft);
        assert_eq!(classify_key(key(KeyCode::Right)), Action::ColumnsRight);
        assert_eq!(classify_key(key(KeyCode::Char('/'))), Action::BeginFilter);
        assert_eq!(classify_key(key(KeyCode::Esc)), Action::ClearFilter);
        assert_eq!(classify_key(key(KeyCode::Char('q'))), Action::Quit);
        assert_eq!(classify_key(key(KeyCode::Char('z'))), Action::Ignore);
    }

    /// A pager that swallows Ctrl-C strands the user.
    #[test]
    fn ctrl_c_quits() {
        let ctrl = |c| KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL);
        assert_eq!(classify_key(ctrl('c')), Action::Quit);
        // A modified key with no binding does nothing rather than falling
        // through to the unmodified meaning.
        assert_eq!(classify_key(ctrl('b')), Action::Ignore);
    }

    /// Enter arrives as CR on some terminals and as LF (Ctrl-J) through
    /// others. Both must page, or the filter prompt cannot be confirmed
    /// and then eats every key that follows.
    #[test]
    fn enter_is_recognised_in_both_of_its_wire_forms() {
        let ctrl = |c| KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL);
        assert!(is_enter(key(KeyCode::Enter)));
        assert!(is_enter(ctrl('j')), "LF must count as Enter");
        assert!(is_enter(ctrl('m')), "CR must count as Enter");
        assert!(!is_enter(key(KeyCode::Char('j'))), "plain j is not Enter");
        assert!(!is_enter(ctrl('c')));

        assert_eq!(classify_key(ctrl('j')), Action::NextPage);
        assert_eq!(classify_key(ctrl('m')), Action::NextPage);
    }

    /// Ctrl-D scrolls in `more` and `less`, so it must not eject the
    /// reader here. It is also emitted unbidden by some pty harnesses,
    /// which would end a session before it began.
    #[test]
    fn ctrl_d_does_not_quit() {
        let ctrl_d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL);
        assert_eq!(classify_key(ctrl_d), Action::Ignore);
    }

    /// Terminals that report key releases must not page twice per press.
    #[test]
    fn key_releases_are_ignored() {
        let mut ev = key(KeyCode::Char(' '));
        ev.kind = KeyEventKind::Release;
        assert_eq!(classify_key(ev), Action::Ignore);
    }

    fn drive(actions: &[Action], page_rows: usize, max_col: usize) -> Viewport {
        let mut v = Viewport::new(page_rows);
        for &a in actions {
            v.apply(a, max_col);
        }
        v
    }

    #[test]
    fn paging_forward_advances_by_a_page() {
        let v = drive(&[Action::NextPage, Action::NextPage], 10, 0);
        assert_eq!(v.top, 20);
    }

    #[test]
    fn paging_back_stops_at_the_first_row() {
        let v = drive(
            &[Action::NextPage, Action::PrevPage, Action::PrevPage],
            10,
            0,
        );
        assert_eq!(v.top, 0);
    }

    /// Back from a non-multiple offset lands on a page boundary, never
    /// below zero.
    #[test]
    fn paging_back_saturates() {
        let mut v = Viewport::new(10);
        v.top = 5;
        assert!(v.apply(Action::PrevPage, 0));
        assert_eq!(v.top, 0);
        // Already home: no redraw.
        assert!(!v.apply(Action::PrevPage, 0));
    }

    #[test]
    fn the_column_window_slides_within_bounds() {
        let mut v = Viewport::new(10);
        assert!(!v.apply(Action::ColumnsLeft, 3), "already leftmost");
        assert!(v.apply(Action::ColumnsRight, 3));
        assert_eq!(v.col_start, 1);
        for _ in 0..10 {
            v.apply(Action::ColumnsRight, 3);
        }
        assert_eq!(v.col_start, 3, "must not scroll past the last window");
        assert!(!v.apply(Action::ColumnsRight, 3));
        assert!(v.apply(Action::ColumnsLeft, 3));
        assert_eq!(v.col_start, 2);
    }

    #[test]
    fn home_returns_to_the_origin() {
        let mut v = drive(&[Action::NextPage, Action::ColumnsRight], 10, 5);
        assert!(v.apply(Action::Home, 5));
        assert_eq!((v.top, v.col_start), (0, 0));
        assert!(!v.apply(Action::Home, 5), "already home");
    }

    /// A short page is how the stream reports its length, and the pager
    /// must not then advance past it.
    #[test]
    fn a_short_page_fixes_the_total_and_stops_forward_paging() {
        let mut v = Viewport::new(10);
        v.top = 20;
        assert!(!v.observe_page(4));
        assert_eq!(v.total_rows, Some(24));
        assert!(!v.apply(Action::NextPage, 0), "must not page past the end");
        assert_eq!(v.top, 20);
    }

    /// A full page says nothing about the total — there may be exactly
    /// zero more rows, and guessing would print a wrong "of N".
    #[test]
    fn a_full_page_leaves_the_total_unknown() {
        let mut v = Viewport::new(10);
        assert!(!v.observe_page(10));
        assert_eq!(v.total_rows, None);
        assert_eq!(v.position(10), "rows 1-10");
    }

    /// Advancing on the last full page can land past the end; the empty
    /// draw that follows must pull the viewport back onto real rows.
    #[test]
    fn scrolling_off_the_end_is_pulled_back() {
        let mut v = Viewport::new(10);
        v.top = 0;
        v.observe_page(10); // exactly 10 rows so far, total unknown
        assert!(v.apply(Action::NextPage, 0));
        assert_eq!(v.top, 10);
        assert!(v.observe_page(0), "an empty page must reposition");
        assert_eq!(v.total_rows, Some(10));
        assert_eq!(v.top, 0, "back to the page that has the rows");
    }

    #[test]
    fn position_reports_the_visible_range() {
        let mut v = Viewport::new(10);
        v.top = 20;
        assert_eq!(v.position(10), "rows 21-30");
        v.observe_page(6);
        assert_eq!(v.position(6), "rows 21-26 of 26");
    }

    /// An empty table is a header and a count, not a crash and not a
    /// blank screen.
    #[test]
    fn an_empty_table_says_zero_rows() {
        let mut v = Viewport::new(10);
        v.observe_page(0);
        assert_eq!(v.total_rows, Some(0));
        assert_eq!(v.position(0), "0 rows");
    }

    /// A filter that matches nothing reads differently from an empty
    /// file — the file has rows, this view does not.
    #[test]
    fn a_filter_with_no_matches_says_so() {
        let mut v = Viewport::new(10);
        v.top = 0;
        assert_eq!(v.position(0), "no matching rows");
        v.observe_page(0);
        assert_eq!(v.position(0), "0 rows");
    }

    /// A pty that reports 0x0 must not collapse the table to one
    /// truncated column.
    #[test]
    fn a_zero_terminal_size_falls_back_to_a_conventional_one() {
        assert_eq!(FALLBACK_SIZE, (80, 24));
        // page_rows is what a zero row count would poison.
        assert_eq!(page_rows(FALLBACK_SIZE.1, true), 21);
    }

    #[test]
    fn page_rows_reserves_room_for_the_chrome() {
        assert_eq!(page_rows(24, true), 21);
        assert_eq!(page_rows(4, true), 1);
        // A terminal too short for even the chrome still gets one row.
        assert_eq!(page_rows(1, true), 1);
        assert_eq!(page_rows(0, true), 1);
    }

    /// `--no-header` hands the header and its rule back to the table
    /// rather than doing nothing, and the status bar moves up with them.
    #[test]
    fn suppressing_the_header_gives_its_two_lines_to_the_rows() {
        assert_eq!(page_rows(24, false), 23);
        assert_eq!(status_row(21, true), 23);
        assert_eq!(status_row(23, false), 23);
    }

    /// A resize changes the page height without losing the reader's place.
    #[test]
    fn a_resize_keeps_the_anchor_row() {
        let mut v = Viewport::new(20);
        v.top = 40;
        v.page_rows = page_rows(10, true);
        assert_eq!(v.page_rows, 7);
        assert_eq!(v.top, 40);
    }
}
