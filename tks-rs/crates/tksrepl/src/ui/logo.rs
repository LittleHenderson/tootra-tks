use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use super::theme::IceBlue;

/// Tootra logo ASCII art - rendered as subtle watermark
pub const TOOTRA_LOGO: &[&str] = &[
    r"                                                              ",
    r"      ████████╗ ██████╗  ██████╗ ████████╗██████╗  █████╗     ",
    r"      ╚══██╔══╝██╔═══██╗██╔═══██╗╚══██╔══╝██╔══██╗██╔══██╗    ",
    r"         ██║   ██║   ██║██║   ██║   ██║   ██████╔╝███████║    ",
    r"         ██║   ██║   ██║██║   ██║   ██║   ██╔══██╗██╔══██║    ",
    r"         ██║   ╚██████╔╝╚██████╔╝   ██║   ██║  ██║██║  ██║    ",
    r"         ╚═╝    ╚═════╝  ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝    ",
    r"                                                              ",
    r"              ╔═══════════════════════════════╗               ",
    r"              ║   THE KABBALISTIC SYSTEM      ║               ",
    r"              ╚═══════════════════════════════╝               ",
    r"                                                              ",
];

/// Render the Tootra logo as a subtle watermark in the background
pub fn render_watermark(area: Rect, buf: &mut Buffer) {
    let logo_height = TOOTRA_LOGO.len() as u16;
    let logo_width = TOOTRA_LOGO.iter().map(|l| l.len()).max().unwrap_or(0) as u16;

    // Don't render if area is too small
    if area.height < logo_height || area.width < logo_width {
        return;
    }

    // Center the logo
    let x = area.x + (area.width.saturating_sub(logo_width)) / 2;
    let y = area.y + (area.height.saturating_sub(logo_height)) / 2;

    let style = IceBlue::watermark();

    for (i, line) in TOOTRA_LOGO.iter().enumerate() {
        let line_y = y + i as u16;
        if line_y >= area.y + area.height {
            break;
        }

        for (j, ch) in line.chars().enumerate() {
            let line_x = x + j as u16;
            if line_x >= area.x + area.width {
                break;
            }

            if let Some(cell) = buf.cell_mut((line_x, line_y)) {
                cell.set_char(ch);
                cell.set_style(style);
            }
        }
    }
}
