"""Tests for section cleaning and the HTML-table -> row-wise text converter."""

from src.utils import clean_section_text, html_table_to_text


def test_th_header_row_becomes_labelled_pairs() -> None:
    html = (
        "<table><caption>Table 1: Incidence</caption>"
        "<tr><th>Adverse reaction</th><th>Drug</th><th>Placebo</th></tr>"
        "<tr><td>Nausea</td><td>25%</td><td>9%</td></tr>"
        "</table>"
    )
    text = html_table_to_text(html)
    assert "Table 1: Incidence" in text
    assert "Adverse reaction: Nausea | Drug: 25% | Placebo: 9%" in text
    assert "<" not in text


def test_table_without_headers_keeps_rows_on_separate_lines() -> None:
    html = "<table><tr><td>GG</td><td>5 to 7 mg</td></tr><tr><td>AA</td><td>3 to 4 mg</td></tr></table>"
    text = html_table_to_text(html)
    assert text.splitlines() == ["GG | 5 to 7 mg", "AA | 3 to 4 mg"]


def test_list_items_inside_a_cell_become_separators() -> None:
    html = (
        "<table><tr><td>adults</td><td><list><item>take 2 tablets</item>"
        "<item>do not exceed 10 tablets</item></list></td></tr></table>"
    )
    text = html_table_to_text(html)
    assert "take 2 tablets; do not exceed 10 tablets" in text
    assert "<item>" not in text


def test_html_entities_are_decoded() -> None:
    html = "<table><tr><td>1,000 mg &#8805; 5%</td></tr></table>"
    assert "≥" in html_table_to_text(html) or ">=" in html_table_to_text(html)


def test_markup_without_rows_falls_back_to_plain_text() -> None:
    text = html_table_to_text("<div>See <b>Section 2</b> for details</div>")
    assert "See Section 2 for details" in text
    assert "<" not in text


def test_clean_section_text_strips_stray_inline_markup() -> None:
    cleaned = clean_section_text("<p>Warnings</p><p>Do not use in pregnancy.</p>", "warnings_and_cautions")
    assert "Do not use in pregnancy." in cleaned
    assert "<" not in cleaned
