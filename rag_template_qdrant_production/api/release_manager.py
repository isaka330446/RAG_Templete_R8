# インデックス世代の登録、切替、ロールバックを管理します。
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from api.config import project_path
from api.vector_store import collection_name_from_settings


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or build_id_now()


class ReleaseManager:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        release_settings = settings.get("release", {})
        self.manifest_path = project_path(release_settings.get("manifest_path", "indexes/release_manifest.json"))
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_corpus_version(self) -> str:
        return str(
            self.settings.get("release", {}).get("default_corpus_version")
            or self.settings.get("answer_cache", {}).get("corpus_version")
            or "default"
        )

    def _default_index_version(self) -> str:
        return str(
            self.settings.get("release", {}).get("default_index_version")
            or self.settings.get("answer_cache", {}).get("index_version")
            or "default"
        )

    def default_release(self) -> dict[str, Any]:
        return {
            "release_id": "default",
            "status": "active",
            "provider": str(self.settings.get("vector_db", {}).get("provider", "chroma")).lower(),
            "collection_name": collection_name_from_settings(self.settings),
            "corpus_version": self._default_corpus_version(),
            "index_version": self._default_index_version(),
            "created_at": None,
            "activated_at": None,
            "row_count": None,
            "embedding_size": None,
            "is_default": True,
        }

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"active_release_id": None, "releases": []}
        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("active_release_id", None)
        data.setdefault("releases", [])
        return data

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        tmp_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        tmp_path.replace(self.manifest_path)

    def list_releases(self) -> list[dict[str, Any]]:
        return self.load_manifest().get("releases", [])

    def get_release(self, release_id: str) -> Optional[dict[str, Any]]:
        for release in self.list_releases():
            if release.get("release_id") == release_id:
                return release
        return None

    def get_active_release(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        active_id = manifest.get("active_release_id")
        for release in manifest.get("releases", []):
            if release.get("release_id") == active_id and release.get("status") == "active":
                return release
        return self.default_release()

    def begin_build(
        self,
        *,
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        timestamp = build_id_now()
        corpus_version = corpus_version or self._default_corpus_version()
        index_version = index_version or timestamp
        base_collection = collection_name_from_settings(self.settings)
        collection_name = collection_name or f"{base_collection}_{slug(index_version)}"
        release_id = f"{slug(corpus_version)}_{slug(index_version)}"

        manifest = self.load_manifest()
        releases = [r for r in manifest.get("releases", []) if r.get("release_id") != release_id]
        release = {
            "release_id": release_id,
            "status": "building",
            "provider": str(self.settings.get("vector_db", {}).get("provider", "chroma")).lower(),
            "collection_name": collection_name,
            "corpus_version": corpus_version,
            "index_version": index_version,
            "created_at": now_utc(),
            "activated_at": None,
            "row_count": None,
            "embedding_size": None,
            "error": None,
        }
        releases.append(release)
        manifest["releases"] = releases
        self.save_manifest(manifest)
        return release

    def complete_build(
        self,
        release_id: str,
        *,
        row_count: int,
        embedding_size: int,
        activate: bool = False,
    ) -> dict[str, Any]:
        manifest = self.load_manifest()
        updated = None
        for release in manifest.get("releases", []):
            if release.get("release_id") == release_id:
                release["row_count"] = row_count
                release["embedding_size"] = embedding_size
                release["status"] = "active" if activate else "staging"
                release["error"] = None
                if activate:
                    release["activated_at"] = now_utc()
                    manifest["active_release_id"] = release_id
                updated = release
            elif activate and release.get("status") == "active":
                release["status"] = "archived"
        if updated is None:
            raise ValueError(f"unknown release_id: {release_id}")
        self.save_manifest(manifest)
        return updated

    def mark_failed(self, release_id: str, error: str) -> None:
        manifest = self.load_manifest()
        for release in manifest.get("releases", []):
            if release.get("release_id") == release_id:
                release["status"] = "failed"
                release["error"] = error
                break
        self.save_manifest(manifest)

    def activate_release(self, release_id: str) -> dict[str, Any]:
        manifest = self.load_manifest()
        target = None
        for release in manifest.get("releases", []):
            if release.get("release_id") == release_id:
                target = release
                break
        if target is None:
            raise ValueError(f"unknown release_id: {release_id}")
        if target.get("status") not in {"staging", "active"}:
            raise ValueError(f"release is not activatable: {target.get('status')}")

        for release in manifest.get("releases", []):
            if release.get("release_id") == release_id:
                release["status"] = "active"
                release["activated_at"] = now_utc()
                target = release
            elif release.get("status") == "active":
                release["status"] = "archived"
        manifest["active_release_id"] = release_id
        self.save_manifest(manifest)
        return target

    def archive_release(self, release_id: str) -> dict[str, Any]:
        manifest = self.load_manifest()
        target = None
        if manifest.get("active_release_id") == release_id:
            raise ValueError("cannot archive the active release")
        for release in manifest.get("releases", []):
            if release.get("release_id") == release_id:
                release["status"] = "archived"
                target = release
                break
        if target is None:
            raise ValueError(f"unknown release_id: {release_id}")
        self.save_manifest(manifest)
        return target
