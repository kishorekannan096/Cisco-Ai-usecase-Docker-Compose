from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable

class CiscoAITheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.cyan,
        neutral_hue: colors.Color | str = colors.slate,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_lg,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)", # Dark gradient
            body_text_color="#f8fafc",
            
            block_background_fill="#1e293b",
            block_label_background_fill="#0f172a",
            block_title_text_color="#38bdf8", # Cyan-ish
            
            button_primary_background_fill="linear-gradient(90deg, #00bceb 0%, #005073 100%)", # Cisco Blue Gradient
            button_primary_text_color="white",
            button_primary_border_color="#005073",
            
            slider_color="#00bceb",
        )

theme = CiscoAITheme()
