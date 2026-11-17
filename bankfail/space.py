import shutil
from pathlib import Path

from bankfail.config import Config
from bankfail.export import check_public, failed_certs

ROOT = Path(__file__).resolve().parents[1]

README = """---
title: FDIC Bank Failure Forecast
emoji: \U0001f3e6
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Which US banks fail within a year, from FDIC call reports
---

Live demo for {repo_url}

It reads precomputed tables from the pipeline in that repo: the backtests, the
2023 check, the stress view and the latest quarter. Banks that are still open
only show up in totals.
"""

DOCKERFILE = """FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

REQUIREMENTS = "altair>=5.0\npandas>=2.2\nstreamlit>=1.52\n"


def build_space(config: Config, log=print) -> Path:
    demo = config.paths.demo_dir
    if not (demo / "panel_summary.json").is_file():
        raise FileNotFoundError(f"{demo} has no demo tables, run the export step first")
    target = config.paths.space_dir
    if target.exists():
        shutil.rmtree(target)
    (target / ".streamlit").mkdir(parents=True)
    shutil.copy2(ROOT / "app.py", target / "app.py")
    shutil.copy2(ROOT / ".streamlit" / "config.toml", target / ".streamlit" / "config.toml")
    shutil.copytree(demo, target / "demo")
    (target / "README.md").write_text(
        README.format(repo_url=config.demo.repo_url), encoding="utf-8"
    )
    (target / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (target / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    check_public(target / "demo", failed_certs(config))
    files = [path for path in target.rglob("*") if path.is_file()]
    size = sum(path.stat().st_size for path in files)
    log(f"space bundle in {target}: {len(files)} files, {size / 1e6:.1f} MB")
    return target
