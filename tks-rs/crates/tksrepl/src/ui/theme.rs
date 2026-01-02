use ratatui::style::{Color, Modifier, Style};

/// Ice Blue theme - Modern intelligence agency aesthetic
pub struct IceBlue;

#[allow(dead_code)]
impl IceBlue {
    // Core colors
    pub const BG_DARK: Color = Color::Rgb(10, 14, 20);       // #0a0e14
    pub const CYAN: Color = Color::Rgb(0, 212, 255);         // #00d4ff
    pub const WHITE: Color = Color::Rgb(230, 237, 243);      // #e6edf3
    pub const DARK_CYAN: Color = Color::Rgb(0, 140, 180);    // Dimmed cyan
    pub const NAVY: Color = Color::Rgb(15, 25, 40);          // Panel bg
    pub const ERROR_RED: Color = Color::Rgb(255, 85, 85);    // #ff5555
    pub const SUCCESS_GREEN: Color = Color::Rgb(85, 255, 127);
    pub const WATERMARK: Color = Color::Rgb(20, 28, 38);     // Very subtle

    // Syntax highlighting colors
    pub const KEYWORD: Color = Color::Rgb(0, 212, 255);      // Cyan
    pub const LITERAL: Color = Color::Rgb(255, 180, 100);    // Orange
    pub const OPERATOR: Color = Color::Rgb(200, 200, 200);   // Light gray
    pub const ELEMENT: Color = Color::Rgb(100, 200, 255);    // Light blue
    pub const NOETIC: Color = Color::Rgb(180, 100, 255);     // Purple
    pub const FOUNDATION: Color = Color::Rgb(255, 200, 100); // Gold
    pub const COMMENT: Color = Color::Rgb(100, 110, 130);    // Gray

    // Pre-built styles
    pub fn prompt() -> Style {
        Style::default().fg(Self::CYAN).add_modifier(Modifier::BOLD)
    }

    pub fn input() -> Style {
        Style::default().fg(Self::WHITE)
    }

    pub fn output() -> Style {
        Style::default().fg(Self::DARK_CYAN)
    }

    pub fn error() -> Style {
        Style::default().fg(Self::ERROR_RED)
    }

    pub fn success() -> Style {
        Style::default().fg(Self::SUCCESS_GREEN)
    }

    pub fn title() -> Style {
        Style::default().fg(Self::CYAN).add_modifier(Modifier::BOLD)
    }

    pub fn border() -> Style {
        Style::default().fg(Self::DARK_CYAN)
    }

    pub fn info() -> Style {
        Style::default().fg(Self::DARK_CYAN).add_modifier(Modifier::ITALIC)
    }

    pub fn watermark() -> Style {
        Style::default().fg(Self::WATERMARK)
    }
}
