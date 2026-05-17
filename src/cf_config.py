#!/usr/bin/env python3
"""
Cloud Foundry Configuration

Parses VCAP_SERVICES to extract credentials for:
- PostgreSQL (pgvector storage)
- Embedding model (from GenAI multi-model service binding)
- Chat model (from GenAI multi-model service binding)
"""

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PG_SERVICE_LABELS = [
    "postgresql", "postgres", "postgresql-db", "elephantsql",
    "neon", "crunchy-postgres", "csb-azure-postgresql", "aws-rds-postgres",
]


def get_vcap_services() -> Dict[str, Any]:
    raw = os.getenv("VCAP_SERVICES", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("VCAP_SERVICES is not valid JSON; returning empty dict")
        return {}


def log_vcap_summary() -> None:
    """
    Log the structure of VCAP_SERVICES (labels, instance names, credential keys)
    without exposing secret values. Call this on startup to aid debugging.
    """
    vcap = get_vcap_services()
    if not vcap:
        logger.warning("VCAP_SERVICES is empty or not set")
        return

    logger.info("VCAP_SERVICES bound services:")
    for label, bindings in vcap.items():
        for b in bindings:
            cred_keys = sorted(b.get("credentials", {}).keys())
            # Show one level deeper for nested dicts (e.g. multi-model plans)
            expanded = {}
            for k, v in b.get("credentials", {}).items():
                if isinstance(v, dict):
                    expanded[k] = sorted(v.keys())
            logger.info(
                "  label=%s  name=%s  plan=%s  cred_keys=%s%s",
                label,
                b.get("name", "?"),
                b.get("plan", "?"),
                cred_keys,
                f"  nested={expanded}" if expanded else "",
            )


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def get_postgres_uri() -> Optional[str]:
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")

    vcap = get_vcap_services()
    for label in _PG_SERVICE_LABELS:
        for binding in vcap.get(label, []):
            creds = binding.get("credentials", {})
            uri = creds.get("uri") or creds.get("url")
            if uri:
                return uri
            if all(k in creds for k in ("hostname", "port", "name", "username", "password")):
                return (
                    f"postgresql://{creds['username']}:{creds['password']}"
                    f"@{creds['hostname']}:{creds['port']}/{creds['name']}"
                )
    return None


# ---------------------------------------------------------------------------
# AI / GenAI model credentials
# ---------------------------------------------------------------------------

def _fetch_advertised_models(config_url: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Query the Tanzu AI Services config_url to discover advertised models.

    Per the multi-model plan binding format, the credential does not contain
    the model name directly. Clients must call config_url to get an up-to-date
    list of model names and their capabilities.

    Reference:
      https://techdocs.broadcom.com/us/en/vmware-tanzu/platform/ai-services/10-3/ai/explanation-single-vs-multi-model-plans.html
    """
    try:
        req = urllib.request.Request(
            config_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("advertisedModels", [])
        logger.info(
            "config_url advertised %d model(s): %s",
            len(models),
            [(m.get("name"), m.get("capabilities")) for m in models],
        )
        return models
    except Exception as exc:
        logger.warning("Failed to fetch advertised models from %s: %s", config_url, exc)
        return []


def get_model_config(model_type: str) -> Optional[Dict[str, Any]]:
    """
    Return normalised credentials for *model_type* ('chat' or 'embedding').

    For Tanzu AI Services multi-model plans the model name is NOT in the
    binding credentials — it is discovered by calling endpoint.config_url
    and matching the capability ('EMBEDDING' or 'CHAT') in advertisedModels.

    Searches every service binding in VCAP_SERVICES, trying multiple credential
    shapes before giving up.
    """
    vcap = get_vcap_services()

    all_bindings: List[Dict[str, Any]] = []
    for bindings in vcap.values():
        all_bindings.extend(bindings)

    capability = model_type.upper()  # "EMBEDDING" or "CHAT"

    for binding in all_bindings:
        creds = binding.get("credentials", {})
        tags = binding.get("tags", [])

        extracted = _extract_model_creds(creds, model_type)
        if extracted:
            normalized = _normalize_model_creds(extracted)
            if normalized.get("api_base"):
                # Tanzu multi-model plan: fetch the real model name from config_url
                config_url = extracted.get("config_url")
                if config_url and normalized.get("api_key"):
                    advertised = _fetch_advertised_models(config_url, normalized["api_key"])
                    for m in advertised:
                        if capability in m.get("capabilities", []):
                            normalized["model_name"] = m["name"]
                            break

                logger.info(
                    "Found %s model in service '%s': api_base=%s model=%s",
                    model_type,
                    binding.get("name", "?"),
                    normalized["api_base"],
                    normalized.get("model_name"),
                )
                return normalized

        if model_type in tags:
            normalized = _normalize_model_creds(creds)
            if normalized.get("api_base"):
                return normalized

    logger.warning(
        "No %s model credentials found. Tried %d service binding(s).",
        model_type, len(all_bindings),
    )
    return None


def _extract_model_creds(creds: Dict[str, Any], model_type: str) -> Optional[Dict[str, Any]]:
    # Pattern A: multi-model plan — {"embedding": {...}, "chat": {...}}
    if model_type in creds and isinstance(creds[model_type], dict):
        section = creds[model_type]
        if any(k in section for k in ("api_base", "url", "api_key", "model", "model_name")):
            return section

    # Pattern B: model_capabilities list — {"model_capabilities": ["embedding", "chat"], ...}
    if model_type in creds.get("model_capabilities", []):
        return creds

    # Pattern C: explicit type field — {"type": "embedding", ...}
    if creds.get("model_capability") == model_type or creds.get("type") == model_type:
        return creds

    # Pattern D: flat credentials with an API endpoint at the top level
    has_endpoint = any(k in creds for k in ("api_base", "base_url", "endpoint", "openai_api_base"))
    has_key = any(k in creds for k in ("api_key", "key", "openai_api_key", "access_key"))
    if has_endpoint and has_key:
        return creds

    # Pattern E: Tanzu ai-models / tanzu-all-models — credentials nested under 'endpoint' key
    #   {"endpoint": {"api_base": "...", "api_key": "...", "name": "...", ...}}
    if "endpoint" in creds and isinstance(creds["endpoint"], dict):
        ep = creds["endpoint"]
        if any(k in ep for k in ("api_base", "openai_api_base", "api_key")):
            return ep

    return None


def _normalize_model_creds(creds: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "api_base": (
            creds.get("api_base")
            or creds.get("openai_api_base")
            or creds.get("url")
            or creds.get("base_url")
            or creds.get("endpoint")
        ),
        "api_key": (
            creds.get("api_key")
            or creds.get("key")
            or creds.get("openai_api_key")
            or creds.get("access_key")
            or "not-needed"
        ),
        # model_name is intentionally left None for Tanzu multi-model plans;
        # it will be populated from config_url in get_model_config().
        "model_name": (
            creds.get("model_name")
            or creds.get("model")
            or creds.get("deployment_name")
            or creds.get("model_id")
        ),
    }
