from pathlib import Path
import json
import os


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_FILE = PROJECT_ROOT / "project_source_and_notebooks.txt"


# پوشه‌هایی که نباید محتوای آن‌ها وارد فایل تکست شود
EXCLUDED_DIRS = {
    ".venv",
    "venv",
    "env",
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "node_modules",

    # دیتای حجیم
    "raw",

    # خروجی‌های باینری/مدل‌ها
    "models",
}


# فایل‌هایی که به دلیل حجم یا باینری بودن نباید خوانده شوند
EXCLUDED_EXTENSIONS = {
    ".nc",
    ".grib",
    ".grb",
    ".h5",
    ".hdf5",
    ".keras",
    ".pkl",
    ".joblib",
    ".npy",
    ".npz",

    ".zip",
    ".tar",
    ".gz",
    ".7z",

    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".ico",

    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",

    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
}


# فایل‌های متنی/کدی که محتوایشان باید ذخیره شود
TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".tsv",
    ".tex",
    ".bib",
    ".sh",
    ".bat",
    ".ps1",
}


# فایل‌های بدون پسوند ولی مهم
IMPORTANT_FILENAMES = {
    "README",
    "LICENSE",
    "Makefile",
    "Dockerfile",
    "Procfile",
    "requirements.txt",
    ".gitignore",
}


# اگر یک فایل متنی خیلی بزرگ شد، بیشتر از این مقدار خوانده نشود
MAX_TEXT_FILE_SIZE_MB = 5

# حداکثر طول یک خروجی متنی Notebook
# برای جلوگیری از نابود شدن فایل به علت printهای چندصدهزار خطی
MAX_NOTEBOOK_OUTPUT_CHARS = 100_000


# ============================================================
# HELPERS
# ============================================================

def is_excluded_path(path: Path) -> bool:
    """
    آیا path داخل یکی از پوشه‌های غیرضروری قرار دارد؟
    """
    relative_parts = path.relative_to(PROJECT_ROOT).parts

    for part in relative_parts:
        if part in EXCLUDED_DIRS:
            return True

    return False


def is_excluded_file(path: Path) -> bool:
    if path == OUTPUT_FILE:
        return True

    if path.name == Path(__file__).name:
        # خود اسکریپت را بعداً می‌توانیم جداگانه هم درج کنیم،
        # ولی برای جلوگیری از recursion اینجا skip می‌شود.
        return True

    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    return False


def human_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]

    size = float(size_bytes)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} TB"


def divider(char="=", length=100):
    return char * length


# ============================================================
# PROJECT TREE
# ============================================================

def build_project_tree() -> str:
    """
    ساختار پروژه را ثبت می‌کند.
    داده‌ها و فایل‌های بزرگ نمایش داده می‌شوند،
    ولی محتواشان خوانده نمی‌شود.
    """

    lines = []

    project_name = PROJECT_ROOT.name
    lines.append(f"{project_name}/")

    def walk(directory: Path, prefix=""):
        try:
            items = sorted(
                directory.iterdir(),
                key=lambda x: (x.is_file(), x.name.lower())
            )
        except PermissionError:
            return

        # فایل خروجی خودمان نمایش داده نشود
        items = [
            item for item in items
            if item != OUTPUT_FILE
        ]

        for index, item in enumerate(items):
            is_last = index == len(items) - 1

            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            if item.is_dir():

                if item.name in EXCLUDED_DIRS:
                    lines.append(
                        f"{prefix}{connector}{item.name}/ [content excluded]"
                    )
                    continue

                lines.append(
                    f"{prefix}{connector}{item.name}/"
                )

                walk(
                    item,
                    prefix + child_prefix
                )

            else:

                try:
                    size = human_size(item.stat().st_size)
                except Exception:
                    size = "unknown"

                if item.suffix.lower() in EXCLUDED_EXTENSIONS:
                    lines.append(
                        f"{prefix}{connector}{item.name} "
                        f"[binary/data, {size}]"
                    )
                else:
                    lines.append(
                        f"{prefix}{connector}{item.name} "
                        f"[{size}]"
                    )

    walk(PROJECT_ROOT)

    return "\n".join(lines)


# ============================================================
# NORMAL TEXT FILES
# ============================================================

def should_read_text_file(path: Path) -> bool:

    if is_excluded_path(path):
        return False

    if is_excluded_file(path):
        return False

    if path.suffix.lower() == ".ipynb":
        return False

    if path.name in IMPORTANT_FILENAMES:
        return True

    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True

    return False


def read_text_file(path: Path) -> str:

    size_mb = path.stat().st_size / (1024 * 1024)

    if size_mb > MAX_TEXT_FILE_SIZE_MB:
        return (
            f"[FILE NOT INCLUDED: size={size_mb:.2f} MB, "
            f"limit={MAX_TEXT_FILE_SIZE_MB} MB]"
        )

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1"
    ]

    for encoding in encodings:
        try:
            return path.read_text(
                encoding=encoding,
                errors="strict"
            )
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            return f"[ERROR READING FILE: {exc}]"

    return "[UNABLE TO DECODE FILE]"


# ============================================================
# NOTEBOOK PARSER
# ============================================================

def normalize_source(source):
    if isinstance(source, list):
        return "".join(source)

    if source is None:
        return ""

    return str(source)


def truncate_output(text: str) -> str:
    if len(text) <= MAX_NOTEBOOK_OUTPUT_CHARS:
        return text

    return (
        text[:MAX_NOTEBOOK_OUTPUT_CHARS]
        + "\n\n"
        + f"[OUTPUT TRUNCATED AFTER "
          f"{MAX_NOTEBOOK_OUTPUT_CHARS:,} CHARACTERS]"
    )


def extract_notebook_output(output) -> str:

    output_type = output.get("output_type", "")

    # print()
    if output_type == "stream":
        text = normalize_source(
            output.get("text", "")
        )

        return truncate_output(text)

    # Exception / traceback
    if output_type == "error":

        ename = output.get("ename", "")
        evalue = output.get("evalue", "")
        traceback = output.get("traceback", [])

        traceback_text = "\n".join(traceback)

        text = (
            f"ERROR TYPE: {ename}\n"
            f"ERROR MESSAGE: {evalue}\n\n"
            f"TRACEBACK:\n{traceback_text}"
        )

        return truncate_output(text)

    # dataframe / expression / matplotlib / display()
    if output_type in {
        "execute_result",
        "display_data"
    }:

        data = output.get("data", {})

        pieces = []

        # بهترین representation برای DataFrame و سایر objectها
        if "text/plain" in data:

            plain = normalize_source(
                data["text/plain"]
            )

            pieces.append(
                "[text/plain]\n" + plain
            )

        # بعضی خروجی‌ها فقط HTML دارند
        if (
            "text/html" in data
            and "text/plain" not in data
        ):

            html = normalize_source(
                data["text/html"]
            )

            pieces.append(
                "[text/html]\n" + html
            )

        # JSON output
        if "application/json" in data:

            try:
                json_data = json.dumps(
                    data["application/json"],
                    ensure_ascii=False,
                    indent=2
                )

                pieces.append(
                    "[application/json]\n"
                    + json_data
                )
            except Exception:
                pass

        # تصاویر را عمداً داخل txt نمی‌ریزیم
        if (
            "image/png" in data
            or "image/jpeg" in data
            or "image/svg+xml" in data
        ):

            pieces.append(
                "[IMAGE OUTPUT OMITTED FROM TEXT EXPORT]"
            )

        if not pieces:
            return "[NON-TEXT NOTEBOOK OUTPUT OMITTED]"

        return truncate_output(
            "\n\n".join(pieces)
        )

    return f"[UNKNOWN OUTPUT TYPE: {output_type}]"


def read_notebook(path: Path) -> str:

    try:
        with path.open(
            "r",
            encoding="utf-8"
        ) as f:
            notebook = json.load(f)

    except Exception as exc:
        return f"[ERROR READING NOTEBOOK: {exc}]"

    sections = []

    cells = notebook.get(
        "cells",
        []
    )

    sections.append(
        f"TOTAL CELLS: {len(cells)}"
    )

    for index, cell in enumerate(
        cells,
        start=1
    ):

        cell_type = cell.get(
            "cell_type",
            "unknown"
        )

        source = normalize_source(
            cell.get(
                "source",
                ""
            )
        )

        sections.append(
            "\n"
            + divider("-", 90)
        )

        sections.append(
            f"CELL {index} | TYPE: {cell_type.upper()}"
        )

        sections.append(
            divider("-", 90)
        )

        if cell_type == "markdown":

            sections.append(
                "\n[MARKDOWN]\n"
            )

            sections.append(
                source
            )

        elif cell_type == "code":

            execution_count = cell.get(
                "execution_count"
            )

            sections.append(
                f"\n[CODE | execution_count={execution_count}]\n"
            )

            sections.append(
                source
            )

            outputs = cell.get(
                "outputs",
                []
            )

            if outputs:

                sections.append(
                    "\n[OUTPUTS]\n"
                )

                for output_index, output in enumerate(
                    outputs,
                    start=1
                ):

                    sections.append(
                        f"\n--- OUTPUT {output_index} ---\n"
                    )

                    sections.append(
                        extract_notebook_output(
                            output
                        )
                    )

            else:

                sections.append(
                    "\n[NO OUTPUT]\n"
                )

        elif cell_type == "raw":

            sections.append(
                "\n[RAW CELL]\n"
            )

            sections.append(
                source
            )

    return "\n".join(sections)


# ============================================================
# EXPORT
# ============================================================

def export_project():

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as out:

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        out.write(
            divider("=")
            + "\n"
        )

        out.write(
            f"PROJECT EXPORT: {PROJECT_ROOT.name}\n"
        )

        out.write(
            f"PROJECT PATH: {PROJECT_ROOT}\n"
        )

        out.write(
            divider("=")
            + "\n\n"
        )

        # ----------------------------------------------------
        # PROJECT STRUCTURE
        # ----------------------------------------------------

        out.write(
            "\n"
            + divider("=")
            + "\n"
        )

        out.write(
            "PROJECT STRUCTURE\n"
        )

        out.write(
            divider("=")
            + "\n\n"
        )

        out.write(
            build_project_tree()
        )

        out.write(
            "\n\n"
        )

        # ----------------------------------------------------
        # NORMAL CODE/TEXT FILES
        # ----------------------------------------------------

        out.write(
            divider("=")
            + "\n"
        )

        out.write(
            "PROJECT SOURCE FILES\n"
        )

        out.write(
            divider("=")
            + "\n"
        )

        all_files = sorted(
            PROJECT_ROOT.rglob("*")
        )

        for path in all_files:

            if not path.is_file():
                continue

            if not should_read_text_file(path):
                continue

            relative_path = path.relative_to(
                PROJECT_ROOT
            )

            out.write(
                "\n\n"
                + divider("=")
                + "\n"
            )

            out.write(
                f"FILE: {relative_path}\n"
            )

            out.write(
                divider("=")
                + "\n\n"
            )

            content = read_text_file(
                path
            )

            out.write(content)

            out.write(
                "\n"
            )

        # ----------------------------------------------------
        # NOTEBOOKS
        # ----------------------------------------------------

        out.write(
            "\n\n"
            + divider("=")
            + "\n"
        )

        out.write(
            "JUPYTER NOTEBOOKS\n"
        )

        out.write(
            divider("=")
            + "\n"
        )

        notebooks = sorted(
            PROJECT_ROOT.rglob(
                "*.ipynb"
            )
        )

        notebooks = [
            nb for nb in notebooks
            if not is_excluded_path(nb)
            and ".ipynb_checkpoints"
            not in nb.parts
        ]

        for notebook in notebooks:

            relative_path = notebook.relative_to(
                PROJECT_ROOT
            )

            out.write(
                "\n\n"
                + "#" * 100
                + "\n"
            )

            out.write(
                f"NOTEBOOK: {relative_path}\n"
            )

            out.write(
                "#" * 100
                + "\n\n"
            )

            out.write(
                read_notebook(
                    notebook
                )
            )

            out.write(
                "\n"
            )

        # ----------------------------------------------------
        # END
        # ----------------------------------------------------

        out.write(
            "\n\n"
            + divider("=")
            + "\n"
        )

        out.write(
            "END OF PROJECT EXPORT\n"
        )

        out.write(
            divider("=")
            + "\n"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    export_project()

    print("\nProject export completed.")
    print("Output file:")
    print(OUTPUT_FILE)

    print(
        "\nFile size:",
        human_size(
            OUTPUT_FILE.stat().st_size
        )
    )