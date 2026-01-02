use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};

use super::app::{App, OutputLine};
use super::logo::render_watermark;
use super::theme::IceBlue;

/// Render the entire UI
pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Fill background with dark color
    let bg_block = Block::default().style(Style::default().bg(IceBlue::BG_DARK));
    frame.render_widget(bg_block, area);

    // Render watermark first (behind everything)
    render_watermark(area, frame.buffer_mut());

    // Main layout: header, output, input
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(5),     // Output
            Constraint::Length(3),  // Input
        ])
        .split(area);

    render_header(frame, chunks[0]);
    render_output(frame, chunks[1], app);
    render_input(frame, chunks[2], app);
}

fn render_header(frame: &mut Frame, area: Rect) {
    let title = " TKS CONSOLE ";
    let subtitle = "[ TOOTRA KABBALISTIC SYSTEM ]";

    let header_text = vec![
        Line::from(vec![
            Span::styled(title, IceBlue::title()),
            Span::raw("  "),
            Span::styled(subtitle, Style::default().fg(IceBlue::DARK_CYAN)),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(IceBlue::border())
        .style(Style::default().bg(IceBlue::NAVY));

    let paragraph = Paragraph::new(header_text)
        .block(block)
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, area);
}

fn render_output(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" Output ")
        .title_style(IceBlue::title())
        .borders(Borders::ALL)
        .border_style(IceBlue::border())
        .style(Style::default().bg(Color::Rgb(12, 16, 22))); // Slightly transparent feel

    let inner_area = block.inner(area);

    // Convert output lines to styled text
    let lines: Vec<Line> = app
        .output
        .iter()
        .map(|line| match line {
            OutputLine::Input(s) => Line::from(vec![
                Span::styled(" > ", IceBlue::prompt()),
                Span::styled(s.clone(), IceBlue::input()),
            ]),
            OutputLine::Result(s) => Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(s.clone(), IceBlue::success()),
            ]),
            OutputLine::Error(s) => Line::from(vec![
                Span::styled(" ! ", IceBlue::error()),
                Span::styled(s.clone(), IceBlue::error()),
            ]),
            OutputLine::Info(s) => Line::from(vec![
                Span::styled(" * ", IceBlue::info()),
                Span::styled(s.clone(), IceBlue::info()),
            ]),
        })
        .collect();

    // Calculate scroll
    let visible_height = inner_area.height as usize;
    let total_lines = lines.len();
    let scroll = if total_lines > visible_height {
        total_lines.saturating_sub(visible_height)
    } else {
        0
    };

    let paragraph = Paragraph::new(lines)
        .block(block)
        .scroll((scroll as u16, 0));

    frame.render_widget(paragraph, area);

    // Render scrollbar if needed
    if total_lines > visible_height {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .style(Style::default().fg(IceBlue::DARK_CYAN));
        let mut scrollbar_state = ScrollbarState::new(total_lines)
            .position(scroll);
        frame.render_stateful_widget(
            scrollbar,
            area.inner(Margin { horizontal: 0, vertical: 1 }),
            &mut scrollbar_state,
        );
    }
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" Input ")
        .title_style(IceBlue::title())
        .borders(Borders::ALL)
        .border_style(IceBlue::border())
        .style(Style::default().bg(IceBlue::NAVY));

    let prompt = "tks> ";
    let input_line = Line::from(vec![
        Span::styled(prompt, IceBlue::prompt()),
        Span::styled(&app.input, IceBlue::input()),
    ]);

    let paragraph = Paragraph::new(input_line).block(block);

    frame.render_widget(paragraph, area);

    // Set cursor position
    let cursor_x = area.x + 1 + prompt.len() as u16 + app.cursor_pos as u16;
    let cursor_y = area.y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));
}
