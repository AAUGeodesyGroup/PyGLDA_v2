import os
import sys
import platform
import unicodedata
from datetime import datetime


def env():
    """
    Checks the MPI environment to determine the rank and size.
    Returns the current process rank (0 for main process, -1 for serial mode).
    """
    mpi_available = False
    mpi_running = False

    try:
        from mpi4py import MPI
        mpi_available = True
        if MPI.Is_initialized():
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            mpi_running = size > 1
        else:
            comm = MPI.COMM_SELF
            size = 1
            rank = -1
            mpi_running = False
    except ImportError:
        mpi_available = False
        comm = None
        size = 1
        rank = -1
        mpi_running = False

    return rank


def print_pyglda_banner(version="2.0.0", model='ReWaterGap 2.2.e'):
    """
    Prints the stylized PyGLDA banner with True Color RGB gradients,
    dynamically calculated borders, perfectly aligned metadata, and MPI status.
    """
    if env() != 0 and env() != -1:
        return

    mpi_size = 1
    try:
        from mpi4py import MPI
        if MPI.Is_initialized():
            mpi_size = MPI.COMM_WORLD.Get_size()
    except ImportError:
        pass

    use_color = True

    def c(text, code):
        return f"\033[{code}m{text}\033[0m" if use_color else text

    def rgb(text, r, g, b):
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m" if use_color else text

    def display_width(text):
        """
        Calculates exact terminal display width, ignoring invisible zero-width modifiers.
        """
        width = 0
        for char in str(text):
            # Ignore zero-width joiners and variation selectors (e.g., U+FE0F)
            if unicodedata.category(char) in ('Mn', 'Me', 'Cf'):
                continue
            if unicodedata.east_asian_width(char) in ('F', 'W'):
                width += 2
            else:
                width += 1
        return width

    # Color definitions
    CYAN = "36"
    GREEN = "32"
    GRAY = "90"
    BOLD = "1"

    top_box = "╭────────────────────────────────────────────────────────────────────────────╮"
    bottom_box = "╰────────────────────────────────────────────────────────────────────────────╯"
    divider = "├────────────────────────────────────────────────────────────────────────────┤"
    empty_line = "│                                                                            │"

    logo_lines = [
        "██████╗  ██╗   ██╗  ██████╗  ██╗       ██████╗    █████╗",
        "██╔══██╗ ╚██╗ ██╔╝ ██╔════╝  ██║       ██╔══██╗  ██╔══██╗",
        "██████╔╝  ╚████╔╝  ██║  ███╗ ██║       ██║  ██║  ███████║",
        "██╔═══╝    ╚██╔╝   ██║   ██║ ██║       ██║  ██║  ██╔══██║",
        "██║         ██║    ╚██████╔╝ ███████╗  ██████╔╝  ██║  ██║",
        "╚═╝         ╚═╝     ╚═════╝  ╚══════╝  ╚═════╝   ╚═╝  ╚═╝"
    ]

    def print_meta_line(icon, name, value, color, value_color=None):
        label = f"{icon} {name}"
        val_str = str(value)

        # Fixed label area: 18 terminal columns
        label_width = display_width(label)
        label_pad = " " * max(0, 18 - label_width)

        # Fixed value area: 51 terminal columns
        val_width = display_width(val_str)
        val_pad = " " * max(0, 51 - val_width)

        formatted_val = c(val_str, value_color) if value_color else val_str

        line_str = f"    {c(label + label_pad, color)} : {formatted_val}{val_pad}"
        print(f"{c('│', CYAN)}{line_str}{c('│', CYAN)}")

    r_start, g_start, b_start = 0, 255, 255
    r_end, g_end, b_end = 0, 80, 255

    print(c(top_box, CYAN))
    print(c(empty_line, CYAN))

    max_logo_width = max(display_width(line) for line in logo_lines)

    for i, line in enumerate(logo_lines):
        line = line.ljust(max_logo_width)

        ratio = i / (len(logo_lines) - 1)
        r = int(r_start + (r_end - r_start) * ratio)
        g = int(g_start + (g_end - g_start) * ratio)
        b = int(b_start + (b_end - b_start) * ratio)
        colored_line = rgb(line, r, g, b)

        left_spaces = (76 - max_logo_width) // 2
        right_spaces = 76 - max_logo_width - left_spaces
        print(f"{c('│', CYAN)}{' ' * left_spaces}{colored_line}{' ' * right_spaces}{c('│', CYAN)}")

    print(c(empty_line, CYAN))

    subtitle = "Python Global Land Data Assimilation System"
    left_pad = (76 - len(subtitle)) // 2
    right_pad = 76 - len(subtitle) - left_pad
    print(f"{c('│', CYAN)}{c(' ' * left_pad + subtitle + ' ' * right_pad, GRAY + ';' + BOLD)}{c('│', CYAN)}")

    print(c(divider, CYAN))
    print(c(empty_line, CYAN))

    # Updated to 100% standard double-width emojis to guarantee flawless left alignment
    print_meta_line("📦", "Version", version, GREEN)
    print_meta_line("🌊", "GHM Model", model, GREEN)
    print_meta_line("🔬", "Institution", "Aalborg University – Geodesy Group", GREEN)
    print_meta_line("🔗", "DOI", "https://doi.org/10.5194/gmd-18-6195-2025", GREEN)
    print_meta_line("📂", "Repository", "github.com/AAUGeodesyGroup/PyGLDA", GREEN)

    print(c(empty_line, CYAN))
    print(c(divider, CYAN))
    print(c(empty_line, CYAN))

    print_meta_line("🕒", "Start Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), GRAY)
    print_meta_line("💻", "Environment", f"Python {platform.python_version()} | {platform.system()}", GRAY)

    if mpi_size > 1:
        mpi_val = f"{mpi_size} Ranks (Parallel Compute Enabled)"
        mpi_val_color = CYAN + ";" + BOLD
    else:
        mpi_val = "1 Rank (Serial Mode)"
        mpi_val_color = GRAY

    print_meta_line("🚀", "Compute Nodes", mpi_val, GRAY, value_color=mpi_val_color)

    print(c(empty_line, CYAN))
    print(c(bottom_box, CYAN))
    print()


if __name__ == "__main__":
    print_pyglda_banner()