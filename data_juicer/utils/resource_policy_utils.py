import os
from typing import Dict, List, Optional

from loguru import logger

from .cache_utils import (
    DATA_JUICER_ASSETS_CACHE,
    DATA_JUICER_EXTERNAL_MODELS_HOME,
    DATA_JUICER_MODELS_CACHE,
)

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


class ResourcePolicyError(ValueError):
    """Raised when a resource policy environment variable is invalid."""


class ResourceResolutionError(RuntimeError):
    """Raised when resource resolution fails under the current policy."""


def _get_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_bool_env(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = _get_env(name)
    if raw is None:
        return default

    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ResourcePolicyError(f"Invalid boolean value for {name}: {raw}")


def _parse_list_env(name: str) -> List[str]:
    raw = _get_env(name)
    if raw is None:
        return []
    return [item.strip() for item in raw.split(os.pathsep) if item.strip()]


def get_resource_policy() -> Dict[str, object]:
    offline_mode = _parse_bool_env("DJ_RESOURCE_OFFLINE_MODE", default=False)
    allow_public_fallback = _parse_bool_env("DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK", default=True)
    hf_local_files_only = _parse_bool_env("DJ_HF_LOCAL_FILES_ONLY", default=None)
    nltk_allow_download = _parse_bool_env("DJ_NLTK_ALLOW_DOWNLOAD", default=True)
    package_auto_install = _parse_bool_env("DJ_PACKAGE_AUTO_INSTALL", default=True)

    if offline_mode:
        allow_public_fallback = False
        package_auto_install = False

    return {
        "offline_mode": offline_mode,
        "allow_public_fallback": allow_public_fallback,
        "local_cache_roots": _parse_list_env("DJ_RESOURCE_LOCAL_CACHE_ROOTS"),
        "model_base_url": _get_env("DJ_MODEL_BASE_URL"),
        "asset_base_url": _get_env("DJ_ASSET_BASE_URL"),
        "hf_endpoint": _get_env("DJ_HF_ENDPOINT"),
        "hf_home": _get_env("DJ_HF_HOME"),
        "hf_local_files_only": hf_local_files_only,
        "nltk_data_dir": _get_env("DJ_NLTK_DATA_DIR"),
        "nltk_allow_download": nltk_allow_download,
        "package_auto_install": package_auto_install,
    }


def should_allow_public_fallback(policy: Optional[Dict[str, object]] = None) -> bool:
    policy = policy or get_resource_policy()
    return bool(policy["allow_public_fallback"]) and not bool(policy["offline_mode"])


def should_auto_install_package(policy: Optional[Dict[str, object]] = None) -> bool:
    policy = policy or get_resource_policy()
    return bool(policy["package_auto_install"]) and not bool(policy["offline_mode"])


def is_nltk_download_allowed(policy: Optional[Dict[str, object]] = None) -> bool:
    policy = policy or get_resource_policy()
    return bool(policy["nltk_allow_download"]) and not bool(policy["offline_mode"])


def _find_local_model_path(model_name: str, policy: Dict[str, object]) -> Optional[Dict[str, str]]:
    if os.path.exists(model_name):
        return {"kind": "local_path", "value": model_name, "source": "explicit_path"}

    cache_path = os.path.join(DATA_JUICER_MODELS_CACHE, model_name)
    if os.path.exists(cache_path):
        return {"kind": "local_path", "value": cache_path, "source": "cache"}

    if DATA_JUICER_EXTERNAL_MODELS_HOME:
        for path in DATA_JUICER_EXTERNAL_MODELS_HOME.split(os.pathsep):
            clean_path = path.strip()
            if not clean_path:
                continue
            external_path = os.path.join(clean_path, model_name)
            if os.path.exists(external_path):
                return {"kind": "local_path", "value": external_path, "source": "external_root"}

    for root in policy["local_cache_roots"]:
        candidate = os.path.join(root, model_name)
        if os.path.exists(candidate):
            return {"kind": "local_path", "value": candidate, "source": "local_cache_root"}

    return None


def resolve_model_source(model_name: str, force: bool = False) -> Dict[str, str]:
    policy = get_resource_policy()
    attempted_sources = []

    if not force:
        local_result = _find_local_model_path(model_name, policy)
        if local_result:
            logger.info(f"Resolved model [{model_name}] from {local_result['source']}: {local_result['value']}")
            return local_result
        attempted_sources.extend(["explicit_path", "cache", "external_root", "local_cache_root"])

    if policy["model_base_url"]:
        mirror_url = os.path.join(str(policy["model_base_url"]).rstrip("/"), model_name)
        logger.info(f"Resolved model [{model_name}] to mirror URL: {mirror_url}")
        return {"kind": "remote_url", "value": mirror_url, "source": "mirror"}
    attempted_sources.append("mirror")

    if should_allow_public_fallback(policy):
        logger.info(f"Resolved model [{model_name}] to default public source")
        return {"kind": "remote_url", "value": model_name, "source": "default_public"}

    raise ResourceResolutionError(
        f"Cannot resolve model [{model_name}] under current policy. attempted_sources={attempted_sources}, "
        f"offline_mode={policy['offline_mode']}, allow_public_fallback={policy['allow_public_fallback']}"
    )


def resolve_asset_source(asset_type: str) -> Dict[str, str]:
    policy = get_resource_policy()
    attempted_sources = []

    cache_path = os.path.join(DATA_JUICER_ASSETS_CACHE, f"{asset_type}.json")
    if os.path.exists(cache_path):
        logger.info(f"Resolved asset [{asset_type}] from cache: {cache_path}")
        return {"kind": "local_path", "value": cache_path, "source": "cache"}
    attempted_sources.append("cache")

    for root in policy["local_cache_roots"]:
        candidate = os.path.join(root, f"{asset_type}.json")
        if os.path.exists(candidate):
            logger.info(f"Resolved asset [{asset_type}] from local cache root: {candidate}")
            return {"kind": "local_path", "value": candidate, "source": "local_cache_root"}
    attempted_sources.append("local_cache_root")

    if policy["asset_base_url"]:
        mirror_url = os.path.join(str(policy["asset_base_url"]).rstrip("/"), f"{asset_type}.json")
        logger.info(f"Resolved asset [{asset_type}] to mirror URL: {mirror_url}")
        return {"kind": "remote_url", "value": mirror_url, "source": "mirror"}
    attempted_sources.append("mirror")

    if should_allow_public_fallback(policy):
        logger.info(f"Resolved asset [{asset_type}] to default public source")
        return {"kind": "remote_url", "value": asset_type, "source": "default_public"}

    raise ResourceResolutionError(
        f"Cannot resolve asset [{asset_type}] under current policy. attempted_sources={attempted_sources}, "
        f"offline_mode={policy['offline_mode']}, allow_public_fallback={policy['allow_public_fallback']}"
    )


def configure_hf_env(policy: Optional[Dict[str, object]] = None) -> None:
    policy = policy or get_resource_policy()

    if policy["hf_endpoint"]:
        os.environ["HF_ENDPOINT"] = str(policy["hf_endpoint"])
    if policy["hf_home"]:
        os.environ["HF_HOME"] = str(policy["hf_home"])

    hf_local_files_only = policy["hf_local_files_only"]
    if hf_local_files_only is True or policy["offline_mode"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def configure_nltk_env(policy: Optional[Dict[str, object]] = None) -> None:
    policy = policy or get_resource_policy()
    nltk_data_dir = policy["nltk_data_dir"]
    if not nltk_data_dir:
        return

    import nltk

    nltk_data_dir = str(nltk_data_dir)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
