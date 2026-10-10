"""Shared CLI pieces: every script takes --profile {smoke,pilot,full}."""

from __future__ import annotations

import argparse
from typing import Any

from mfsi_cxr.utils.config import PROFILES, load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=None,
        help="scale profile (default: env MFSI_PROFILE, else pilot)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.profile)
