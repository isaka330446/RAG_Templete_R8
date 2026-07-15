# 本番向けインデックスの世代登録、切替、ロールバックをCLIで実行します。
from pathlib import Path
import argparse
import json

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.config import load_settings
from api.release_manager import ReleaseManager


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect or switch RAG index releases.")
    parser.add_argument("--list", action="store_true", help="List known releases.")
    parser.add_argument("--active", action="store_true", help="Show the active release.")
    parser.add_argument("--activate", metavar="RELEASE_ID", help="Activate a staging release.")
    parser.add_argument("--archive", metavar="RELEASE_ID", help="Archive a non-active release.")
    return parser.parse_args()


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    manager = ReleaseManager(load_settings())

    if args.activate:
        print_json(manager.activate_release(args.activate))
        return
    if args.archive:
        print_json(manager.archive_release(args.archive))
        return
    if args.active:
        print_json(manager.get_active_release())
        return

    print_json(manager.list_releases())


if __name__ == "__main__":
    main()
