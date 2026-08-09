//! Turning cells into aligned text.
//!
//! Three concerns live here, all pure and all unit-tested: how a number
//! is written, how wide a column is allowed to get, and which slice of a
//! very wide table fits on screen.

use super::source::Cell;

/// How floating-point cells are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatFormat {
    /// Six significant digits (C's `%.6g`). Readable, and enough to see
    /// the physics; not enough to reconstruct the file.
    Significant(usize),
    /// Rust's shortest representation that round-trips to the same `f64`.
    /// Exact — what you read is what is in the file.
    Full,
}

impl Default for FloatFormat {
    fn default() -> Self {
        Self::Significant(6)
    }
}

/// The widest a column may grow before its cells are truncated.
///
/// Wide enough for a full-precision double (`-1.2345678901234567e-102`)
/// and an ISO timestamp; narrow enough that one prose column cannot push
/// every number off the screen.
pub const MAX_COLUMN_WIDTH: usize = 26;

/// Two spaces between columns, as `column -t` uses.
pub const COLUMN_GAP: usize = 2;

/// Render a cell.
///
/// Null and NaN are visually distinct: an absent value is an empty cell,
/// a not-a-number value says `NaN`. In a residuals table those mean
/// different things — no star catalogue was recorded, versus a χ² that
/// could not be evaluated — and one must never read as the other.
pub fn render_cell(cell: &Cell, format: FloatFormat) -> String {
    match cell {
        Cell::Null => String::new(),
        Cell::Bool(b) => b.to_string(),
        Cell::Int(i) => i.to_string(),
        Cell::UInt(u) => u.to_string(),
        Cell::Text(s) => s.clone(),
        Cell::Float(v) => render_float(*v, format),
    }
}

/// Write a float.
pub fn render_float(v: f64, format: FloatFormat) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    match format {
        FloatFormat::Full => full(v),
        FloatFormat::Significant(digits) => significant(v, digits),
    }
}

/// Shortest round-tripping form — Rust's `Display` for `f64` already
/// guarantees this, so `--full-precision` loses nothing.
fn full(v: f64) -> String {
    let s = v.to_string();
    // `to_string` writes integral values as `60000`; keep the trailing
    // `.0` so a float column never looks like an integer column.
    if s.contains(['.', 'e', 'E', 'n', 'i']) {
        s
    } else {
        format!("{s}.0")
    }
}

/// C's `%g` with `digits` significant figures: fixed notation in the
/// readable range, scientific outside it, with trailing zeros trimmed.
///
/// The exponent that picks the notation is the one the value has *after*
/// rounding to `digits` figures, which is why it is read back out of a
/// scientific formatting rather than computed with `log10`. Rounding
/// 999999.9 to six figures carries it to 1000000, and only the
/// post-rounding exponent knows that.
fn significant(v: f64, digits: usize) -> String {
    let digits = digits.max(1);
    if v == 0.0 {
        // Preserve the sign of negative zero; it is a real distinction in
        // a residual column.
        return if v.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let mantissa_places = digits - 1;
    let scientific = format!("{v:.mantissa_places$e}");
    let exp: i32 = scientific
        .split_once('e')
        .and_then(|(_, e)| e.parse().ok())
        .unwrap_or(0);

    // C's rule: style `e` when the exponent is below -4 or at least the
    // precision.
    if exp < -4 || exp >= digits as i32 {
        trim_scientific(&scientific)
    } else {
        let places = (digits as i32 - 1 - exp).max(0) as usize;
        trim_fixed(&format!("{v:.places$}"))
    }
}

/// Drop trailing zeros (and a bare trailing point) from fixed notation.
fn trim_fixed(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    t.strip_suffix('.').unwrap_or(t).to_string()
}

/// Drop trailing mantissa zeros in scientific notation, and write the
/// exponent with an explicit sign so `1e-12` and `1e+12` line up.
fn trim_scientific(s: &str) -> String {
    let Some((mantissa, exp)) = s.split_once('e') else {
        return s.to_string();
    };
    let mantissa = trim_fixed(mantissa);
    if exp.starts_with('-') {
        format!("{mantissa}e{exp}")
    } else {
        format!("{mantissa}e+{exp}")
    }
}

/// Fit `text` into `width` columns, marking any loss with an ellipsis.
///
/// Truncation is always signalled. A silently clipped number is a wrong
/// number, so the `…` is what tells the reader to widen the terminal,
/// narrow the selection with `--columns`, or pipe the table out.
pub fn fit(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len <= width {
        return format!("{text:<width$}");
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_string();
    }
    let kept: String = text.chars().take(width - 1).collect();
    format!("{kept}…")
}

/// Column widths that only ever grow.
///
/// A streaming pager cannot know the widest cell in a column without
/// reading the whole file, which is exactly what it refuses to do. So
/// widths are learned from the rows actually displayed — and never
/// shrink, so the table settles as you page instead of twitching every
/// time a narrow page goes by.
#[derive(Debug, Clone, Default)]
pub struct Widths {
    widths: Vec<usize>,
}

impl Widths {
    /// Seed from the header, which is always visible and therefore always
    /// a lower bound.
    pub fn from_header(header: &[String]) -> Self {
        Self {
            widths: header
                .iter()
                .map(|h| h.chars().count().min(MAX_COLUMN_WIDTH))
                .collect(),
        }
    }

    /// Widen to accommodate a rendered row.
    pub fn observe(&mut self, rendered: &[String]) {
        for (w, cell) in self.widths.iter_mut().zip(rendered) {
            *w = (*w).max(cell.chars().count()).min(MAX_COLUMN_WIDTH);
        }
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.widths
    }
}

/// Which columns fit on screen, starting at `start`.
///
/// Returns the half-open range of column indices to draw. At least one
/// column is always returned — a terminal narrower than the first column
/// shows that column truncated rather than an empty screen.
pub fn visible_columns(widths: &[usize], start: usize, terminal_width: usize) -> (usize, usize) {
    if widths.is_empty() {
        return (0, 0);
    }
    let start = start.min(widths.len() - 1);
    let mut used = 0_usize;
    let mut end = start;
    for (i, w) in widths.iter().enumerate().skip(start) {
        let needed = if i == start { *w } else { COLUMN_GAP + *w };
        if used + needed > terminal_width && i > start {
            break;
        }
        used += needed;
        end = i + 1;
    }
    (start, end.max(start + 1))
}

/// The furthest left a column window may start and still fill the screen.
///
/// Scrolling right past this would leave a mostly blank screen with the
/// last column stranded at the left margin, so `→` stops here.
pub fn max_column_start(widths: &[usize], terminal_width: usize) -> usize {
    if widths.is_empty() {
        return 0;
    }
    let last = widths.len() - 1;
    let mut start = last;
    let mut used = widths[last];
    while start > 0 {
        let candidate = used + COLUMN_GAP + widths[start - 1];
        if candidate > terminal_width {
            break;
        }
        used = candidate;
        start -= 1;
    }
    start
}

/// Draw one row of already-rendered cells over the visible column window.
pub fn draw_row(rendered: &[String], widths: &[usize], range: (usize, usize)) -> String {
    let (start, end) = range;
    let mut out = String::new();
    for i in start..end.min(rendered.len()) {
        if i > start {
            out.push_str(&" ".repeat(COLUMN_GAP));
        }
        out.push_str(&fit(&rendered[i], widths[i]));
    }
    // The padding on the final column is invisible and only makes lines
    // hard to diff in a test or a pipe.
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig6(v: f64) -> String {
        render_float(v, FloatFormat::Significant(6))
    }

    #[test]
    fn six_significant_digits_matches_printf_g() {
        // Cross-checked against C `printf("%.6g")`.
        assert_eq!(sig6(0.0), "0");
        assert_eq!(sig6(1.0), "1");
        assert_eq!(sig6(60000.0), "60000");
        assert_eq!(sig6(1.0 / 3.0), "0.333333");
        assert_eq!(sig6(123456.0), "123456");
        assert_eq!(sig6(1234567.0), "1.23457e+6");
        assert_eq!(sig6(1e-12), "1e-12");
        assert_eq!(sig6(0.0001), "0.0001");
        assert_eq!(sig6(0.00001), "1e-5");
        assert_eq!(sig6(-std::f64::consts::PI), "-3.14159");
        assert_eq!(sig6(2.5e-13), "2.5e-13");
    }

    /// Rounding at the significant-digit boundary must carry, not clip.
    #[test]
    fn significant_digits_round_rather_than_truncate() {
        assert_eq!(sig6(0.99999999), "1");
        assert_eq!(sig6(1.2345675), "1.23457");
        assert_eq!(sig6(9.999999e5), "1e+6");
    }

    /// Full precision is exact: every value must survive the round trip,
    /// which is the entire promise of `--full-precision`.
    #[test]
    fn full_precision_round_trips_exactly() {
        for v in [
            1.0 / 3.0,
            std::f64::consts::PI,
            1e-300,
            2.2250738585072014e-308,
            60000.123456789,
            -1.7976931348623157e308,
            0.1 + 0.2,
        ] {
            let s = render_float(v, FloatFormat::Full);
            let back: f64 = s.parse().expect("full precision must be parseable");
            assert_eq!(
                back.to_bits(),
                v.to_bits(),
                "--full-precision must be lossless, got {s} for {v:e}"
            );
        }
    }

    /// A float column must not masquerade as an integer column.
    #[test]
    fn full_precision_keeps_a_decimal_point_on_integral_values() {
        assert_eq!(render_float(60000.0, FloatFormat::Full), "60000.0");
        assert_eq!(render_float(-1.0, FloatFormat::Full), "-1.0");
    }

    #[test]
    fn non_finite_values_are_named_in_both_modes() {
        for f in [FloatFormat::Significant(6), FloatFormat::Full] {
            assert_eq!(render_float(f64::NAN, f), "NaN");
            assert_eq!(render_float(f64::INFINITY, f), "inf");
            assert_eq!(render_float(f64::NEG_INFINITY, f), "-inf");
        }
    }

    /// Null and NaN mean different things and must look different.
    #[test]
    fn null_renders_empty_and_nan_renders_nan() {
        let f = FloatFormat::default();
        assert_eq!(render_cell(&Cell::Null, f), "");
        assert_eq!(render_cell(&Cell::Float(f64::NAN), f), "NaN");
    }

    #[test]
    fn negative_zero_keeps_its_sign() {
        assert_eq!(sig6(-0.0), "-0");
        assert_eq!(sig6(0.0), "0");
    }

    #[test]
    fn fit_pads_short_text_and_ellipsizes_long_text() {
        assert_eq!(fit("ab", 5), "ab   ");
        assert_eq!(fit("abcde", 5), "abcde");
        assert_eq!(fit("abcdef", 5), "abcd…");
        assert_eq!(fit("abc", 1), "…");
        assert_eq!(fit("abc", 0), "");
        assert_eq!(fit("", 3), "   ");
    }

    /// Truncation counts characters, not bytes — a multibyte designation
    /// must not be cut mid-character.
    #[test]
    fn fit_truncates_on_character_boundaries() {
        assert_eq!(fit("Encke–Faye", 6), "Encke…");
        assert_eq!(fit("αβγδε", 4), "αβγ…");
        assert_eq!(fit("αβγ", 5), "αβγ  ");
    }

    fn header(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn widths_seed_from_the_header_and_only_grow() {
        let mut w = Widths::from_header(&header(&["a", "bbbb"]));
        assert_eq!(w.as_slice(), &[1, 4]);
        w.observe(&header(&["xxx", "y"]));
        assert_eq!(w.as_slice(), &[3, 4], "widths must never shrink");
        w.observe(&header(&["x", "y"]));
        assert_eq!(w.as_slice(), &[3, 4]);
    }

    #[test]
    fn widths_are_capped() {
        let mut w = Widths::from_header(&header(&["a"]));
        w.observe(&["x".repeat(500)]);
        assert_eq!(w.as_slice(), &[MAX_COLUMN_WIDTH]);
    }

    /// The header itself is capped too, or a pathological column name
    /// would consume the screen before a single value was drawn.
    #[test]
    fn a_very_long_header_is_capped() {
        let w = Widths::from_header(&["h".repeat(500)]);
        assert_eq!(w.as_slice(), &[MAX_COLUMN_WIDTH]);
    }

    #[test]
    fn visible_columns_fills_the_terminal() {
        let widths = [10, 10, 10, 10];
        // 10 + 2+10 = 22 fits in 24; adding another needs 34.
        assert_eq!(visible_columns(&widths, 0, 24), (0, 2));
        assert_eq!(visible_columns(&widths, 0, 34), (0, 3));
        assert_eq!(visible_columns(&widths, 0, 1000), (0, 4));
    }

    #[test]
    fn visible_columns_starts_where_asked() {
        let widths = [10, 10, 10, 10];
        assert_eq!(visible_columns(&widths, 2, 24), (2, 4));
        assert_eq!(visible_columns(&widths, 3, 24), (3, 4));
    }

    /// A window that cannot fit even one column still shows one, so the
    /// screen is never blank.
    #[test]
    fn at_least_one_column_is_always_visible() {
        let widths = [40, 40];
        assert_eq!(visible_columns(&widths, 0, 5), (0, 1));
        assert_eq!(visible_columns(&[], 0, 80), (0, 0));
    }

    /// Asking to start beyond the last column clamps instead of panicking.
    #[test]
    fn visible_columns_clamps_an_out_of_range_start() {
        let widths = [10, 10];
        assert_eq!(visible_columns(&widths, 99, 80), (1, 2));
    }

    #[test]
    fn max_column_start_leaves_a_full_screen() {
        let widths = [10, 10, 10, 10];
        // 24 columns holds two 10-wide columns, so the rightmost window
        // starts at index 2.
        assert_eq!(max_column_start(&widths, 24), 2);
        assert_eq!(max_column_start(&widths, 1000), 0);
        assert_eq!(max_column_start(&widths, 5), 3);
        assert_eq!(max_column_start(&[], 80), 0);
    }

    /// Scrolling right to the limit and reading the window back must
    /// include the last column — otherwise a column is unreachable.
    #[test]
    fn the_last_column_is_always_reachable() {
        for term in [5, 12, 24, 37, 80] {
            let widths = [10, 12, 8, 20, 6];
            let start = max_column_start(&widths, term);
            let (_, end) = visible_columns(&widths, start, term);
            assert_eq!(end, widths.len(), "term={term} start={start}");
        }
    }

    #[test]
    fn draw_row_aligns_and_trims_the_trailing_pad() {
        let cells = header(&["a", "bb", "c"]);
        let widths = [4, 4, 4];
        assert_eq!(draw_row(&cells, &widths, (0, 3)), "a     bb    c");
        assert_eq!(draw_row(&cells, &widths, (1, 3)), "bb    c");
    }

    #[test]
    fn draw_row_ellipsizes_cells_wider_than_their_column() {
        let cells = header(&["abcdefgh"]);
        assert_eq!(draw_row(&cells, &[4], (0, 1)), "abc…");
    }
}
